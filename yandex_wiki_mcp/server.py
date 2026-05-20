from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any
from urllib.parse import urlparse

import backoff
import httpx
from pydantic import Field, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.context import Context
from mcp.types import ToolAnnotations

from importlib.metadata import version

__version__ = version("mcp-yandex-wiki")


@asynccontextmanager
async def _lifespan(server: FastMCP):
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield {"http_client": client}


mcp = FastMCP(
    "yandex-wiki",
    version=__version__,
    instructions=(
        "This server provides access to Yandex Wiki (wiki.yandex.ru). "
        "When the user sends a link like https://wiki.yandex.ru/..., "
        "use wiki_page_get_text_by_url or wiki_page_get_by_url to retrieve the page content. "
        "Do NOT use WebFetch or WebSearch for wiki.yandex.ru — they will fail authentication."
    ),
    lifespan=_lifespan,
)

DEFAULT_FIELDS = "content,attributes,breadcrumbs,redirect"
HTTP_TRANSPORTS = {"http", "streamable-http", "sse"}
SERVER_READONLY = False
TOOLS_CACHE_PREFIX = "yandex_wiki_mcp:tools_cache:v2json"
_CACHE_INDEX_ADAPTER = TypeAdapter(list[str])


class _RuntimeEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    wiki_token: str | None = Field(default=None, validation_alias="WIKI_TOKEN")
    tracker_token: str | None = Field(default=None, validation_alias="TRACKER_TOKEN")
    wiki_org_id: str | None = Field(default=None, validation_alias="WIKI_ORG_ID")
    tracker_org_id: str | None = Field(default=None, validation_alias="TRACKER_ORG_ID")
    wiki_api_base_url: str = Field(
        default="https://api.wiki.yandex.net/v1",
        validation_alias="WIKI_API_BASE_URL",
    )


class _ToolsCacheEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    enabled: bool = Field(default=False, validation_alias="TOOLS_CACHE_ENABLED")
    redis_endpoint: str = Field(default="localhost", validation_alias="REDIS_ENDPOINT")
    redis_port: int = Field(default=6379, ge=1, validation_alias="REDIS_PORT")
    redis_db: int = Field(default=0, ge=0, validation_alias="REDIS_DB")
    redis_password: str | None = Field(default=None, validation_alias="REDIS_PASSWORD")
    redis_pool_max_size: int = Field(default=10, ge=1, validation_alias="REDIS_POOL_MAX_SIZE")
    redis_ttl: int = Field(default=3600, ge=0, validation_alias="TOOLS_CACHE_REDIS_TTL")


class _ReadonlyEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    readonly: bool = Field(default=False, validation_alias="READONLY")


def _runtime_settings() -> tuple[str, str, str]:
    env = _RuntimeEnv()
    token = env.wiki_token or env.tracker_token
    org_id = env.wiki_org_id or env.tracker_org_id
    base_url = env.wiki_api_base_url
    return token or "", org_id or "", base_url


def _authorization_header(token: str) -> str:
    value = token.strip()
    lowered = value.lower()
    if lowered.startswith("oauth ") or lowered.startswith("bearer "):
        return value
    return f"OAuth {value}"


def _require_env() -> tuple[str, str, str]:
    token, org_id, base_url = _runtime_settings()
    missing = []
    if not token:
        missing.append("WIKI_TOKEN (или TRACKER_TOKEN)")
    if not org_id:
        missing.append("WIKI_ORG_ID (или TRACKER_ORG_ID)")
    if missing:
        raise RuntimeError("Не заданы переменные окружения: " + ", ".join(missing))
    return token, org_id, base_url


def _normalize_slug(slug: str) -> str:
    normalized = (slug or "").strip()
    normalized = normalized.lstrip("/")
    normalized = normalized.rstrip("/")
    return normalized


def _slug_from_full_url(full_url: str) -> str:
    parsed_url = urlparse(full_url)
    return _normalize_slug(parsed_url.path or "")


def _normalize_fields(fields: str) -> str:
    raw_fields = (fields or "").strip()
    if not raw_fields:
        return DEFAULT_FIELDS
    if raw_fields.lower() in {"text", "body"}:
        return "content"

    parts = [part.strip() for part in raw_fields.split(",") if part.strip()]
    return ",".join(parts) if parts else DEFAULT_FIELDS


def _error_response(status_code: int, error: str) -> dict:
    return {"ok": False, "status_code": status_code, "error": error}


def _normalize_page_id(page_id: int) -> int:
    try:
        normalized = int(page_id)
    except (TypeError, ValueError):
        raise ToolError("Параметр page_id должен быть целым числом.")
    if normalized <= 0:
        raise ToolError("Параметр page_id должен быть положительным целым числом.")
    return normalized


def _normalize_grid_id(grid_id: str) -> str:
    if grid_id is None:
        raise ToolError("Параметр grid_id не должен быть пустым.")
    normalized = str(grid_id).strip()
    if not normalized:
        raise ToolError("Параметр grid_id не должен быть пустым.")
    return normalized


def _normalize_required_str(value: Any, param_name: str) -> str:
    if value is None:
        raise ToolError(f"Параметр {param_name} не должен быть пустым.")
    normalized = str(value).strip()
    if not normalized:
        raise ToolError(f"Параметр {param_name} не должен быть пустым.")
    return normalized


def _bool_param(value: bool) -> str:
    return str(bool(value)).lower()


def _drop_none(params: dict) -> dict:
    return {key: value for key, value in params.items() if value is not None}


def _assert_write_enabled(tool_name: str):
    if SERVER_READONLY:
        raise ToolError(f"Инструмент '{tool_name}' отключен: сервер запущен в режиме readonly.")


def _get_http_client(ctx: Context) -> httpx.AsyncClient:
    client = (ctx.lifespan_context or {}).get("http_client")
    if client is None:
        raise ToolError("HTTP-клиент недоступен: lifespan не инициализирован.")
    return client


def _build_tools_cache() -> tuple[Any | None, int]:
    try:
        settings = _ToolsCacheEnv()
    except ValidationError as exc:
        raise RuntimeError("Некорректные значения переменных окружения кэша:") from exc

    if not settings.enabled:
        return None, 0

    try:
        from aiocache import Cache
        from aiocache.serializers import JsonSerializer
    except ImportError as exc:
        raise RuntimeError(
            "Для TOOLS_CACHE_ENABLED=true требуется зависимость aiocache[redis].",
        ) from exc

    class _RedisJsonSerializer(JsonSerializer):
        def dumps(self, value):
            if not isinstance(value, (dict, list)):
                return json.dumps(value)
            return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    cache = Cache(
        Cache.REDIS,
        endpoint=settings.redis_endpoint,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        pool_max_size=settings.redis_pool_max_size,
        serializer=_RedisJsonSerializer(),
    )
    ttl = settings.redis_ttl
    return cache, ttl


TOOLS_CACHE, TOOLS_CACHE_TTL = _build_tools_cache()


def _cache_ttl_or_none() -> int | None:
    return TOOLS_CACHE_TTL if TOOLS_CACHE_TTL > 0 else None


def _cache_key_for_get(path: str, params: dict | None = None) -> str:
    serialized_params = json.dumps(
        params or {},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = f"{path}|{serialized_params}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"{TOOLS_CACHE_PREFIX}:get:{digest}"


def _cache_slug_index_key(slug: str) -> str:
    return f"{TOOLS_CACHE_PREFIX}:index:slug:{_normalize_slug(slug)}"


def _cache_page_index_key(page_id: int) -> str:
    return f"{TOOLS_CACHE_PREFIX}:index:page:{page_id}"


def _cache_page_slug_mapping_key(page_id: int) -> str:
    return f"{TOOLS_CACHE_PREFIX}:mapping:page_slug:{page_id}"


def _is_error_result(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("ok") is False


def _with_cache_hit(payload: Any, *, cache_hit: bool) -> Any:
    if not isinstance(payload, dict):
        return payload
    result = dict(payload)
    result["_mcp_cache_hit"] = cache_hit
    return result


def _extract_page_id(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None

    candidate = payload.get("id")
    if candidate is None:
        return None

    try:
        parsed = int(candidate)
    except (TypeError, ValueError):
        return None

    return parsed if parsed > 0 else None


def _extract_page_slug(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    raw_slug = payload.get("slug")
    if not isinstance(raw_slug, str):
        return None

    normalized = _normalize_slug(raw_slug)
    return normalized or None


def _validate_cache_index(raw_value: Any) -> list[str] | None:
    try:
        values = _CACHE_INDEX_ADAPTER.validate_python(raw_value)
    except ValidationError:
        return None

    if any(not isinstance(item, str) or not item for item in values):
        return None

    return values


def _validate_cached_payload(raw_value: Any) -> Any | None:
    try:
        json.dumps(
            raw_value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return None
    return raw_value


def _validate_cached_slug(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str):
        return None
    normalized = _normalize_slug(raw_value)
    return normalized or None


async def _cache_index_add(index_key: str, cache_key: str):
    if TOOLS_CACHE is None:
        return

    existing_keys = await TOOLS_CACHE.get(index_key)
    validated_keys = _validate_cache_index(existing_keys)
    if existing_keys is not None and validated_keys is None:
        await TOOLS_CACHE.delete(index_key)
        validated_keys = []
    if validated_keys is None:
        existing_keys = []
    else:
        existing_keys = validated_keys

    if cache_key not in existing_keys:
        existing_keys.append(cache_key)
        await TOOLS_CACHE.set(index_key, existing_keys, ttl=_cache_ttl_or_none())


async def _cache_invalidate_index(index_key: str):
    if TOOLS_CACHE is None:
        return

    existing_keys = await TOOLS_CACHE.get(index_key)
    validated_keys = _validate_cache_index(existing_keys)
    if existing_keys is not None and validated_keys is None:
        await TOOLS_CACHE.delete(index_key)
        return

    if validated_keys is not None:
        for cache_key in set(validated_keys):
            await TOOLS_CACHE.delete(cache_key)

    await TOOLS_CACHE.delete(index_key)


async def _cache_link_page(page_id: int, slug: str):
    if TOOLS_CACHE is None:
        return
    normalized_slug = _normalize_slug(slug)
    if not normalized_slug:
        return
    await TOOLS_CACHE.set(
        _cache_page_slug_mapping_key(page_id),
        normalized_slug,
        ttl=_cache_ttl_or_none(),
    )


async def _cache_register_page_entry(cache_key: str, request_slug: str | None, response_payload: Any):
    if TOOLS_CACHE is None or _is_error_result(response_payload):
        return

    normalized_request_slug = _normalize_slug(request_slug or "")
    response_slug = _extract_page_slug(response_payload)
    page_id = _extract_page_id(response_payload)

    slugs_to_index: set[str] = set()
    if normalized_request_slug:
        slugs_to_index.add(normalized_request_slug)
    if response_slug:
        slugs_to_index.add(response_slug)

    for slug in slugs_to_index:
        await _cache_index_add(_cache_slug_index_key(slug), cache_key)

    if page_id is not None:
        await _cache_index_add(_cache_page_index_key(page_id), cache_key)
        if response_slug:
            await _cache_link_page(page_id, response_slug)
        elif normalized_request_slug:
            await _cache_link_page(page_id, normalized_request_slug)


async def _invalidate_page_cache(page_id: int | None = None, slug: str | None = None):
    if TOOLS_CACHE is None:
        return

    normalized_slug = _normalize_slug(slug or "")

    if page_id is not None:
        await _cache_invalidate_index(_cache_page_index_key(page_id))
        mapping_key = _cache_page_slug_mapping_key(page_id)
        mapped_slug_raw = await TOOLS_CACHE.get(mapping_key)
        mapped_slug = _validate_cached_slug(mapped_slug_raw)
        if mapped_slug_raw is not None and mapped_slug is None:
            await TOOLS_CACHE.delete(mapping_key)
        await TOOLS_CACHE.delete(_cache_page_slug_mapping_key(page_id))

        if not normalized_slug and isinstance(mapped_slug, str):
            normalized_slug = _normalize_slug(mapped_slug)

    if normalized_slug:
        await _cache_invalidate_index(_cache_slug_index_key(normalized_slug))


async def _request(
    method: str,
    path: str,
    params: dict | None = None,
    body: dict | None = None,
    *,
    http_client: httpx.AsyncClient,
):
    try:
        token, org_id, base_url = _require_env()
    except RuntimeError as exc:
        return _error_response(400, str(exc))

    headers = {
        "Authorization": _authorization_header(token),
        "X-Org-Id": org_id,
    }
    request_url = f"{base_url}{path}"

    @backoff.on_exception(
        backoff.expo,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
        ),
        max_tries=4,
        max_time=30,
        giveup=lambda e: isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.status_code < 500,
    )
    async def _send_request() -> httpx.Response:
        client = http_client
        response = await client.request(
            method=method,
            url=request_url,
            headers=headers,
            params=params,
            json=body,
        )
        response.raise_for_status()
        return response

    try:
        response = await _send_request()
    except httpx.TimeoutException:
        return {
            "ok": False,
            "status_code": 504,
            "url": request_url,
            "error": "Таймаут при обращении к API Yandex Wiki.",
        }
    except httpx.HTTPStatusError as exc:
        response = exc.response
        if response is None:
            return {
                "ok": False,
                "status_code": 502,
                "url": request_url,
                "error": f"Ошибка HTTP при обращении к API Yandex Wiki: {exc}",
            }
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        return {
            "ok": False,
            "status_code": response.status_code,
            "url": str(response.request.url),
            "response": payload,
        }
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "status_code": 502,
            "url": request_url,
            "error": f"Ошибка HTTP при обращении к API Yandex Wiki: {exc}",
        }

    if response.status_code >= 400:
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        return {
            "ok": False,
            "status_code": response.status_code,
            "url": str(response.request.url),
            "response": payload,
        }

    try:
        return response.json()
    except Exception:
        return {
            "ok": True,
            "status_code": response.status_code,
            "url": str(response.request.url),
            "response": response.text,
        }


async def _request_get(
    path: str,
    params: dict | None = None,
    *,
    cache_slug: str | None = None,
    http_client: httpx.AsyncClient,
) -> Any:
    if TOOLS_CACHE is None:
        payload = await _request(method="GET", path=path, params=params, http_client=http_client)
        return _with_cache_hit(payload, cache_hit=False)

    cache_key = _cache_key_for_get(path=path, params=params)
    raw_cached_payload = await TOOLS_CACHE.get(cache_key)
    cached_payload = _validate_cached_payload(raw_cached_payload)
    if raw_cached_payload is not None and cached_payload is None:
        await TOOLS_CACHE.delete(cache_key)
    if cached_payload is not None:
        await _cache_register_page_entry(
            cache_key=cache_key,
            request_slug=cache_slug,
            response_payload=cached_payload,
        )
        return _with_cache_hit(cached_payload, cache_hit=True)

    payload = await _request(method="GET", path=path, params=params, http_client=http_client)
    if _is_error_result(payload):
        return _with_cache_hit(payload, cache_hit=False)

    validated_payload = _validate_cached_payload(payload)
    if validated_payload is None:
        return _error_response(500, "Получен некорректный формат ответа для кэширования.")

    await TOOLS_CACHE.set(cache_key, validated_payload, ttl=_cache_ttl_or_none())
    await _cache_register_page_entry(
        cache_key=cache_key,
        request_slug=cache_slug,
        response_payload=validated_payload,
    )
    return _with_cache_hit(validated_payload, cache_hit=False)


async def _get_page_by_slug(
    slug: str,
    fields: str,
    raise_on_redirect: bool = False,
    *,
    http_client: httpx.AsyncClient,
) -> dict:
    normalized_slug = _normalize_slug(slug)
    if not normalized_slug:
        raise ToolError("Параметр slug не должен быть пустым.")

    params = {
        "slug": normalized_slug,
        "fields": _normalize_fields(fields),
        "raise_on_redirect": str(bool(raise_on_redirect)).lower(),
    }
    return await _request_get(path="/pages", params=params, cache_slug=normalized_slug, http_client=http_client)


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_get_by_url(
    url: Annotated[str, Field(description="Полная ссылка на страницу, например https://wiki.yandex.ru/users/handbook/")],
    ctx: Context,
    fields: str = Field(default=DEFAULT_FIELDS, description="Поля через запятую: content, attributes, breadcrumbs, redirect"),
    raise_on_redirect: bool = Field(default=False, description="Вернуть ошибку при редиректе вместо автоматического перехода"),
) -> dict:
    """Read-only: получить страницу по полной ссылке вида https://wiki.yandex.ru/<path...>/"""
    slug = _slug_from_full_url(url)
    await ctx.info(f"Запрашиваю страницу по URL: {url} (slug={slug})")
    http_client = _get_http_client(ctx)
    return await _get_page_by_slug(slug=slug, fields=fields, raise_on_redirect=raise_on_redirect, http_client=http_client)


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_get(
    slug: Annotated[str, Field(description="Путь страницы без домена, например 'users/handbook/onboarding'")],
    ctx: Context,
    fields: str = Field(default=DEFAULT_FIELDS, description="Поля через запятую: content, attributes, breadcrumbs, redirect"),
    raise_on_redirect: bool = Field(default=False, description="Вернуть ошибку при редиректе вместо автоматического перехода"),
) -> dict:
    """Read-only: получить страницу по slug (путь без домена)."""
    await ctx.info(f"Запрашиваю страницу: {slug}")
    http_client = _get_http_client(ctx)
    return await _get_page_by_slug(slug=slug, fields=fields, raise_on_redirect=raise_on_redirect, http_client=http_client)


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_get_text_by_url(
    url: Annotated[str, Field(description="Полная ссылка на страницу, например https://wiki.yandex.ru/users/handbook/")],
    ctx: Context,
) -> dict:
    """Read-only: вернуть только content страницы по полной ссылке."""
    slug = _slug_from_full_url(url)
    await ctx.info(f"Запрашиваю текст страницы по URL: {url} (slug={slug})")
    http_client = _get_http_client(ctx)
    data = await _get_page_by_slug(slug=slug, fields="content", http_client=http_client)
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    result = {"ok": True, "content": data.get("content")}
    if isinstance(data, dict) and isinstance(data.get("_mcp_cache_hit"), bool):
        result["_mcp_cache_hit"] = data["_mcp_cache_hit"]
    return result


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_resolve_id(
    ctx: Context,
    slug: str | None = Field(default=None, description="Путь страницы без домена, например 'users/handbook/onboarding'"),
    url: str | None = Field(default=None, description="Полная ссылка на страницу, например https://wiki.yandex.ru/users/handbook/"),
) -> dict:
    """Read-only: получить page_id страницы по slug или полной ссылке. Используйте перед wiki_page_update / wiki_page_append_content, если известен только slug или URL."""
    if url and not slug:
        slug = _slug_from_full_url(url)
    if not slug or not slug.strip():
        raise ToolError("Необходимо указать slug или url.")

    normalized_slug = _normalize_slug(slug)
    if not normalized_slug:
        raise ToolError("Параметр slug не должен быть пустым.")

    await ctx.info(f"Резолвлю page_id для slug={normalized_slug}")
    http_client = _get_http_client(ctx)
    data = await _get_page_by_slug(slug=normalized_slug, fields="attributes", http_client=http_client)

    if _is_error_result(data):
        return data

    resolved_id = _extract_page_id(data)
    if resolved_id is None:
        raise ToolError(f"Не удалось определить page_id для slug '{normalized_slug}'.")

    result = {"ok": True, "page_id": resolved_id, "slug": normalized_slug}
    if isinstance(data, dict) and isinstance(data.get("_mcp_cache_hit"), bool):
        result["_mcp_cache_hit"] = data["_mcp_cache_hit"]
    return result


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_page_create(
    slug: Annotated[str, Field(description="Путь новой страницы без домена, например 'users/handbook/new-page'")],
    title: Annotated[str, Field(description="Заголовок страницы")],
    content: Annotated[str, Field(description="Содержимое страницы в формате Wiki/WYSIWYG")],
    ctx: Context,
    page_type: str = Field(default="wysiwyg", description="Тип страницы: wysiwyg или wikitext"),
    fields: str = Field(default=DEFAULT_FIELDS, description="Поля в ответе через запятую: content, attributes, breadcrumbs, redirect"),
    is_silent: bool = Field(default=False, description="Не отправлять уведомления подписчикам"),
) -> dict:
    """Write: создать новую страницу."""
    await ctx.info(f"Создаю страницу: {slug} (title={title!r})")
    _assert_write_enabled("wiki_page_create")

    normalized_slug = _normalize_slug(slug)
    if not normalized_slug:
        raise ToolError("Параметр slug не должен быть пустым.")
    if not (title or "").strip():
        raise ToolError("Параметр title не должен быть пустым.")

    http_client = _get_http_client(ctx)
    body = {
        "page_type": page_type,
        "slug": normalized_slug,
        "title": title.strip(),
        "content": content,
    }
    params = {
        "fields": _normalize_fields(fields),
        "is_silent": str(bool(is_silent)).lower(),
    }
    result = await _request(method="POST", path="/pages", params=params, body=body, http_client=http_client)
    if not _is_error_result(result):
        await _invalidate_page_cache(
            page_id=_extract_page_id(result),
            slug=_extract_page_slug(result) or normalized_slug,
        )
    return result


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_update(
    page_id: Annotated[int, Field(description="Числовой ID страницы для обновления")],
    ctx: Context,
    title: str | None = Field(default=None, description="Новый заголовок страницы (None — не менять)"),
    content: str | None = Field(default=None, description="Новое содержимое страницы (None — не менять)"),
    allow_merge: bool = Field(default=False, description="Разрешить слияние при конфликте версий"),
    fields: str = Field(default=DEFAULT_FIELDS, description="Поля в ответе через запятую: content, attributes, breadcrumbs, redirect"),
    is_silent: bool = Field(default=False, description="Не отправлять уведомления подписчикам"),
) -> dict:
    """Write: обновить существующую страницу по ID (заголовок и/или контент)."""
    await ctx.info(f"Обновляю страницу ID={page_id}")
    _assert_write_enabled("wiki_page_update")
    normalized_page_id = _normalize_page_id(page_id)

    body = {}
    if title is not None:
        stripped_title = title.strip()
        if not stripped_title:
            raise ToolError("Если title передан, он не должен быть пустым.")
        body["title"] = stripped_title
    if content is not None:
        body["content"] = content
    if not body:
        raise ToolError("Нужно передать хотя бы одно поле для обновления: title или content.")

    http_client = _get_http_client(ctx)
    params = {
        "allow_merge": str(bool(allow_merge)).lower(),
        "fields": _normalize_fields(fields),
        "is_silent": str(bool(is_silent)).lower(),
    }
    result = await _request(method="POST", path=f"/pages/{normalized_page_id}", params=params, body=body, http_client=http_client)
    if not _is_error_result(result):
        await _invalidate_page_cache(
            page_id=normalized_page_id,
            slug=_extract_page_slug(result),
        )
    return result


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_page_append_content(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    content: Annotated[str, Field(description="Содержимое для добавления")],
    ctx: Context,
    location: str = Field(default="bottom", description="Позиция вставки: top, bottom или якорь в формате #anchor"),
    fields: str = Field(default=DEFAULT_FIELDS, description="Поля в ответе через запятую: content, attributes, breadcrumbs, redirect"),
    is_silent: bool = Field(default=False, description="Не отправлять уведомления подписчикам"),
) -> dict:
    """Write: добавить контент в начало/конец страницы или по якорю (#anchor)."""
    await ctx.info(f"Добавляю контент к странице ID={page_id} (location={location})")
    _assert_write_enabled("wiki_page_append_content")
    normalized_page_id = _normalize_page_id(page_id)

    if not (content or "").strip():
        raise ToolError("Параметр content не должен быть пустым.")

    body = {"content": content}
    normalized_location = (location or "").strip()
    if normalized_location.lower() in {"top", "bottom", ""}:
        body["body"] = {"location": (normalized_location or "bottom").lower()}
    elif normalized_location.startswith("#"):
        body["anchor"] = {"name": normalized_location}
    else:
        raise ToolError("Параметр location должен быть top, bottom или якорем в формате #anchor.")

    http_client = _get_http_client(ctx)
    params = {
        "fields": _normalize_fields(fields),
        "is_silent": str(bool(is_silent)).lower(),
    }
    result = await _request(
        method="POST",
        path=f"/pages/{normalized_page_id}/append-content",
        params=params,
        body=body,
        http_client=http_client,
    )
    if not _is_error_result(result):
        await _invalidate_page_cache(
            page_id=normalized_page_id,
            slug=_extract_page_slug(result),
        )
    return result


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_get_by_id(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    ctx: Context,
    fields: str = Field(default=DEFAULT_FIELDS, description="Поля через запятую: content, attributes, breadcrumbs, redirect"),
    raise_on_redirect: bool = Field(default=False, description="Вернуть ошибку при редиректе вместо автоматического перехода"),
    revision_id: int | None = Field(default=None, description="ID конкретной ревизии страницы"),
) -> dict:
    """Read-only: получить страницу по числовому ID."""
    normalized_page_id = _normalize_page_id(page_id)
    await ctx.info(f"Запрашиваю страницу по ID: {normalized_page_id}")
    http_client = _get_http_client(ctx)
    params = _drop_none(
        {
            "fields": _normalize_fields(fields),
            "raise_on_redirect": _bool_param(raise_on_redirect),
            "revision_id": revision_id,
        }
    )
    return await _request(
        method="GET",
        path=f"/pages/{normalized_page_id}",
        params=params,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
)
async def wiki_page_delete(
    page_id: Annotated[int, Field(description="Числовой ID страницы для удаления")],
    ctx: Context,
) -> dict:
    """Write: удалить страницу по ID. В ответе возвращается recovery_token для восстановления."""
    await ctx.info(f"Удаляю страницу ID={page_id}")
    _assert_write_enabled("wiki_page_delete")
    normalized_page_id = _normalize_page_id(page_id)

    http_client = _get_http_client(ctx)
    result = await _request(
        method="DELETE",
        path=f"/pages/{normalized_page_id}",
        http_client=http_client,
    )
    if not _is_error_result(result):
        await _invalidate_page_cache(page_id=normalized_page_id)
    return result


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_page_clone(
    page_id: Annotated[int, Field(description="Числовой ID исходной страницы")],
    target: Annotated[str, Field(description="Slug новой страницы (адрес после клонирования)")],
    ctx: Context,
    title: str | None = Field(default=None, description="Если задан — заголовок новой страницы"),
    subscribe_me: bool = Field(default=False, description="Подписаться на изменения новой страницы"),
) -> dict:
    """Write: клонировать страницу по новому адресу. Асинхронная операция: возвращается task_id для wiki_operation_clone_status."""
    await ctx.info(f"Клонирую страницу ID={page_id} -> {target}")
    _assert_write_enabled("wiki_page_clone")
    normalized_page_id = _normalize_page_id(page_id)
    normalized_target = _normalize_slug(_normalize_required_str(target, "target"))

    body: dict[str, Any] = {
        "target": normalized_target,
        "subscribe_me": bool(subscribe_me),
    }
    if title is not None:
        stripped_title = title.strip()
        if not stripped_title:
            raise ToolError("Если title передан, он не должен быть пустым.")
        body["title"] = stripped_title

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/pages/{normalized_page_id}/clone",
        body=body,
        http_client=http_client,
    )


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_descendants(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    ctx: Context,
    cursor: str | None = Field(default=None, description="Курсор пагинации"),
    actuality: str | None = Field(default=None, description="Фильтр актуальности: actual или obsolete"),
    include_self: bool = Field(default=False, description="Включать саму страницу в результат"),
    page_size: int = Field(default=50, ge=1, le=100, description="Размер страницы (1-100)"),
) -> dict:
    """Read-only: получить все подстраницы (на любом уровне) указанной страницы по ID."""
    normalized_page_id = _normalize_page_id(page_id)
    await ctx.info(f"Получаю подстраницы для ID={normalized_page_id}")
    http_client = _get_http_client(ctx)
    params = _drop_none(
        {
            "cursor": cursor,
            "actuality": actuality,
            "include_self": _bool_param(include_self),
            "page_size": page_size,
        }
    )
    return await _request(
        method="GET",
        path=f"/pages/{normalized_page_id}/descendants",
        params=params,
        http_client=http_client,
    )


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_descendants_by_slug(
    slug: Annotated[str, Field(description="Путь страницы без домена")],
    ctx: Context,
    cursor: str | None = Field(default=None, description="Курсор пагинации"),
    actuality: str | None = Field(default=None, description="Фильтр актуальности: actual или obsolete"),
    include_self: bool = Field(default=False, description="Включать саму страницу в результат"),
    page_size: int = Field(default=50, ge=1, le=100, description="Размер страницы (1-100)"),
) -> dict:
    """Read-only: получить подстраницы по slug страницы."""
    normalized_slug = _normalize_slug(_normalize_required_str(slug, "slug"))
    await ctx.info(f"Получаю подстраницы для slug={normalized_slug}")
    http_client = _get_http_client(ctx)
    params = _drop_none(
        {
            "slug": normalized_slug,
            "cursor": cursor,
            "actuality": actuality,
            "include_self": _bool_param(include_self),
            "page_size": page_size,
        }
    )
    return await _request(
        method="GET",
        path="/pages/descendants",
        params=params,
        http_client=http_client,
    )

@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_attachments_list(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    ctx: Context,
    cursor: str | None = Field(default=None, description="Курсор пагинации"),
    order_by: str | None = Field(default=None, description="Поле сортировки: name, size или created_at"),
    order_direction: str | None = Field(default=None, description="Направление сортировки: asc или desc"),
    page_size: int = Field(default=50, ge=1, le=100, description="Размер страницы (1-100)"),
) -> dict:
    """Read-only: список вложений (файлов), прикреплённых к странице."""
    normalized_page_id = _normalize_page_id(page_id)
    await ctx.info(f"Получаю вложения страницы ID={normalized_page_id}")
    http_client = _get_http_client(ctx)
    params = _drop_none(
        {
            "cursor": cursor,
            "order_by": order_by,
            "order_direction": order_direction,
            "page_size": page_size,
        }
    )
    return await _request(
        method="GET",
        path=f"/pages/{normalized_page_id}/attachments",
        params=params,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_attachment_attach(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    upload_sessions: Annotated[list[str], Field(description="Список ID завершённых upload-сессий, файлы которых нужно прикрепить")],
    ctx: Context,
) -> dict:
    """Write: прикрепить к странице файлы из завершённых upload-сессий."""
    await ctx.info(f"Прикрепляю файлы к странице ID={page_id}")
    _assert_write_enabled("wiki_attachment_attach")
    normalized_page_id = _normalize_page_id(page_id)

    if not upload_sessions or not isinstance(upload_sessions, list):
        raise ToolError("Параметр upload_sessions должен содержать хотя бы один session_id.")
    cleaned_sessions: list[str] = []
    for session in upload_sessions:
        cleaned = _normalize_required_str(session, "upload_sessions[*]")
        cleaned_sessions.append(cleaned)

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/pages/{normalized_page_id}/attachments",
        body={"upload_sessions": cleaned_sessions},
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
async def wiki_attachment_delete(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    file_id: Annotated[int, Field(description="Числовой ID файла-вложения")],
    ctx: Context,
) -> dict:
    """Write: удалить вложение по ID файла со страницы."""
    await ctx.info(f"Удаляю вложение file_id={file_id} со страницы ID={page_id}")
    _assert_write_enabled("wiki_attachment_delete")
    normalized_page_id = _normalize_page_id(page_id)
    try:
        normalized_file_id = int(file_id)
    except (TypeError, ValueError):
        raise ToolError("Параметр file_id должен быть целым числом.")
    if normalized_file_id <= 0:
        raise ToolError("Параметр file_id должен быть положительным целым числом.")

    http_client = _get_http_client(ctx)
    return await _request(
        method="DELETE",
        path=f"/pages/{normalized_page_id}/attachments/{normalized_file_id}",
        http_client=http_client,
    )


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_page_resources_list(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    ctx: Context,
    cursor: str | None = Field(default=None, description="Курсор пагинации"),
    order_by: str | None = Field(default=None, description="Поле сортировки: name_title или created_at"),
    order_direction: str | None = Field(default=None, description="Направление сортировки: asc или desc"),
    page_size: int = Field(default=50, ge=1, le=100, description="Размер страницы (1-100)"),
    q: str | None = Field(default=None, description="Поиск по заголовку (макс. 255 символов)"),
    types: str | None = Field(default=None, description="Типы ресурсов через запятую: attachment, grid"),
) -> dict:
    """Read-only: получить список ресурсов страницы (attachment, grid и пр.)."""
    normalized_page_id = _normalize_page_id(page_id)
    await ctx.info(f"Получаю ресурсы страницы ID={normalized_page_id}")
    http_client = _get_http_client(ctx)
    params = _drop_none(
        {
            "cursor": cursor,
            "order_by": order_by,
            "order_direction": order_direction,
            "page_size": page_size,
            "q": q,
            "types": types,
        }
    )
    return await _request(
        method="GET",
        path=f"/pages/{normalized_page_id}/resources",
        params=params,
        http_client=http_client,
    )
@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_comments_list(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    ctx: Context,
    cursor: str | None = Field(default=None, description="Курсор пагинации"),
    order_by: str | None = Field(default=None, description="Поле сортировки (поддерживается только created_at)"),
    order_direction: str | None = Field(default=None, description="Направление сортировки: asc или desc"),
    page_size: int = Field(default=50, ge=1, le=100, description="Размер страницы (1-100)"),
    status_filter: str | None = Field(default=None, description="Фильтр по статусу: resolved или unresolved"),
) -> dict:
    """Read-only: список комментариев страницы (треды верхнего уровня)."""
    normalized_page_id = _normalize_page_id(page_id)
    await ctx.info(f"Получаю комментарии страницы ID={normalized_page_id}")
    http_client = _get_http_client(ctx)
    params = _drop_none(
        {
            "cursor": cursor,
            "order_by": order_by,
            "order_direction": order_direction,
            "page_size": page_size,
            "status_filter": status_filter,
        }
    )
    return await _request(
        method="GET",
        path=f"/pages/{normalized_page_id}/comments",
        params=params,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_comment_create(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    body: Annotated[str, Field(description="Текст комментария")],
    ctx: Context,
    inline_text: str | None = Field(default=None, description="Текст, к которому привязан inline-комментарий"),
    parent_id: int | None = Field(default=None, description="ID комментария-родителя (для ответа)"),
    thread_id: int | None = Field(default=None, description="ID треда комментария"),
) -> dict:
    """Write: создать комментарий на странице (или ответ в треде)."""
    await ctx.info(f"Создаю комментарий на странице ID={page_id}")
    _assert_write_enabled("wiki_comment_create")
    normalized_page_id = _normalize_page_id(page_id)
    normalized_body = _normalize_required_str(body, "body")

    request_body: dict[str, Any] = {"body": normalized_body}
    if inline_text is not None:
        request_body["inline_text"] = inline_text
    if parent_id is not None:
        request_body["parent_id"] = int(parent_id)
    if thread_id is not None:
        request_body["thread_id"] = int(thread_id)

    http_client = _get_http_client(ctx)
    result = await _request(
        method="POST",
        path=f"/pages/{normalized_page_id}/comments",
        body=request_body,
        http_client=http_client,
    )
    if not _is_error_result(result):
        await _invalidate_page_cache(page_id=normalized_page_id)
    return result


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_comment_thread(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    comment_id: Annotated[int, Field(description="ID корневого комментария треда")],
    ctx: Context,
    cursor: str | None = Field(default=None, description="Курсор пагинации"),
    page_size: int = Field(default=25, ge=1, le=50, description="Размер страницы (1-50)"),
) -> dict:
    """Read-only: получить комментарии в треде по его корневому комментарию."""
    normalized_page_id = _normalize_page_id(page_id)
    normalized_comment_id = int(comment_id)
    if normalized_comment_id <= 0:
        raise ToolError("Параметр comment_id должен быть положительным целым числом.")

    await ctx.info(f"Получаю тред комментария ID={normalized_comment_id} (page_id={normalized_page_id})")
    http_client = _get_http_client(ctx)
    params = _drop_none({"cursor": cursor, "page_size": page_size})
    return await _request(
        method="GET",
        path=f"/pages/{normalized_page_id}/comments/{normalized_comment_id}/thread",
        params=params,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
async def wiki_comment_delete(
    page_id: Annotated[int, Field(description="Числовой ID страницы")],
    comment_id: Annotated[int, Field(description="ID комментария для удаления")],
    ctx: Context,
) -> dict:
    """Write: удалить комментарий со страницы."""
    await ctx.info(f"Удаляю комментарий ID={comment_id} (page_id={page_id})")
    _assert_write_enabled("wiki_comment_delete")
    normalized_page_id = _normalize_page_id(page_id)
    normalized_comment_id = int(comment_id)
    if normalized_comment_id <= 0:
        raise ToolError("Параметр comment_id должен быть положительным целым числом.")

    http_client = _get_http_client(ctx)
    result = await _request(
        method="DELETE",
        path=f"/pages/{normalized_page_id}/comments/{normalized_comment_id}",
        http_client=http_client,
    )
    if not _is_error_result(result):
        await _invalidate_page_cache(page_id=normalized_page_id)
    return result

mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_grid_create(
    title: Annotated[str, Field(description="Заголовок новой динамической таблицы (1-255 символов)")],
    ctx: Context,
    page_id: int | None = Field(default=None, description="ID страницы, к которой будет привязана таблица"),
    page_slug: str | None = Field(default=None, description="Slug страницы (используется если page_id не задан)"),
) -> dict:
    """Write: создать новую динамическую таблицу (grid) как ресурс страницы."""
    await ctx.info(f"Создаю динамическую таблицу title={title!r}")
    _assert_write_enabled("wiki_grid_create")

    normalized_title = _normalize_required_str(title, "title")
    if len(normalized_title) > 255:
        raise ToolError("Параметр title должен быть не длиннее 255 символов.")

    if page_id is None and not (page_slug or "").strip():
        raise ToolError("Нужно указать page_id или page_slug.")

    page_identity: dict[str, Any] = {}
    if page_id is not None:
        page_identity["id"] = _normalize_page_id(page_id)
    if page_slug:
        normalized_slug = _normalize_slug(page_slug)
        if normalized_slug:
            page_identity["slug"] = normalized_slug

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path="/grids",
        body={"title": normalized_title, "page": page_identity},
        http_client=http_client,
    )


@mcp.tool(
    tags={"read", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
async def wiki_grid_get(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    ctx: Context,
    fields: str | None = Field(default=None, description="Дополнительные поля через запятую (attributes, user_permissions, ...)"),
    filter: str | None = Field(default=None, description="Фильтр строк, например '[slug] ~ wiki AND [slug2]<32'"),
    only_cols: str | None = Field(default=None, description="Слаги колонок через запятую — возвращать только их"),
    only_rows: str | None = Field(default=None, description="ID строк через запятую — возвращать только их"),
    revision: int | None = Field(default=None, description="Загрузить старую ревизию таблицы"),
    sort: str | None = Field(default=None, description="Сортировка строк по колонкам, например 'slug, -slug2, slug3'"),
) -> dict:
    """Read-only: получить динамическую таблицу со структурой и строками."""
    normalized_grid_id = _normalize_grid_id(grid_id)
    await ctx.info(f"Получаю grid ID={normalized_grid_id}")
    http_client = _get_http_client(ctx)
    params = _drop_none(
        {
            "fields": fields,
            "filter": filter,
            "only_cols": only_cols,
            "only_rows": only_rows,
            "revision": revision,
            "sort": sort,
        }
    )
    return await _request(
        method="GET",
        path=f"/grids/{normalized_grid_id}",
        params=params,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_grid_update(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    revision: Annotated[str, Field(description="Текущая ревизия таблицы (для оптимистической блокировки)")],
    ctx: Context,
    title: str | None = Field(default=None, description="Новый заголовок таблицы (1-255 символов)"),
    default_sort: list[dict] | None = Field(default=None, description="Сортировка по умолчанию: список объектов {slug, direction}"),
) -> dict:
    """Write: обновить мета-данные динамической таблицы (заголовок и/или сортировку по умолчанию)."""
    await ctx.info(f"Обновляю grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_update")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_revision = _normalize_required_str(revision, "revision")

    body: dict[str, Any] = {"revision": normalized_revision}
    if title is not None:
        stripped_title = title.strip()
        if not stripped_title:
            raise ToolError("Если title передан, он не должен быть пустым.")
        if len(stripped_title) > 255:
            raise ToolError("Параметр title должен быть не длиннее 255 символов.")
        body["title"] = stripped_title
    if default_sort is not None:
        body["default_sort"] = default_sort

    if len(body) == 1:
        raise ToolError("Нужно передать хотя бы одно поле для обновления: title или default_sort.")

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/grids/{normalized_grid_id}",
        body=body,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
async def wiki_grid_delete(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    ctx: Context,
) -> dict:
    """Write: удалить динамическую таблицу."""
    await ctx.info(f"Удаляю grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_delete")
    normalized_grid_id = _normalize_grid_id(grid_id)

    http_client = _get_http_client(ctx)
    return await _request(
        method="DELETE",
        path=f"/grids/{normalized_grid_id}",
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_grid_add_rows(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    revision: Annotated[str, Field(description="Текущая ревизия таблицы")],
    rows: Annotated[list[list[Any]], Field(description="Список строк (каждая строка — массив значений по колонкам)")],
    ctx: Context,
    position: int | None = Field(default=None, description="Позиция вставки (0-based индекс)"),
    after_row_id: str | None = Field(default=None, description="ID строки, после которой вставить новые"),
) -> dict:
    """Write: добавить строки в динамическую таблицу. Значения колонок передаются массивами."""
    await ctx.info(f"Добавляю {len(rows)} строк в grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_add_rows")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_revision = _normalize_required_str(revision, "revision")

    if not rows:
        raise ToolError("Параметр rows должен содержать хотя бы одну строку.")

    body: dict[str, Any] = {"revision": normalized_revision, "rows": rows}
    if position is not None:
        body["position"] = int(position)
    if after_row_id is not None:
        body["after_row_id"] = _normalize_required_str(after_row_id, "after_row_id")

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/grids/{normalized_grid_id}/rows",
        body=body,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
async def wiki_grid_remove_rows(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    revision: Annotated[str, Field(description="Текущая ревизия таблицы")],
    row_ids: Annotated[list[str], Field(description="ID строк для удаления (минимум 1)")],
    ctx: Context,
) -> dict:
    """Write: удалить строки из динамической таблицы."""
    await ctx.info(f"Удаляю строки из grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_remove_rows")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_revision = _normalize_required_str(revision, "revision")

    if not row_ids:
        raise ToolError("Параметр row_ids должен содержать хотя бы один ID.")
    cleaned_row_ids = [_normalize_required_str(row_id, "row_ids[*]") for row_id in row_ids]

    http_client = _get_http_client(ctx)
    return await _request(
        method="DELETE",
        path=f"/grids/{normalized_grid_id}/rows",
        body={"revision": normalized_revision, "row_ids": cleaned_row_ids},
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_grid_add_columns(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    revision: Annotated[str, Field(description="Текущая ревизия таблицы")],
    columns: Annotated[list[dict], Field(description="Описание колонок (title, type, slug, и т.д.)")],
    ctx: Context,
    position: int | None = Field(default=None, description="Позиция вставки колонок"),
) -> dict:
    """Write: добавить колонки в динамическую таблицу."""
    await ctx.info(f"Добавляю {len(columns)} колонок в grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_add_columns")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_revision = _normalize_required_str(revision, "revision")

    if not columns:
        raise ToolError("Параметр columns должен содержать хотя бы одну колонку.")

    body: dict[str, Any] = {"revision": normalized_revision, "columns": columns}
    if position is not None:
        body["position"] = int(position)

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/grids/{normalized_grid_id}/columns",
        body=body,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
async def wiki_grid_remove_columns(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    revision: Annotated[str, Field(description="Текущая ревизия таблицы")],
    column_slugs: Annotated[list[str], Field(description="Слаги колонок для удаления")],
    ctx: Context,
) -> dict:
    """Write: удалить колонки из динамической таблицы по слагам."""
    await ctx.info(f"Удаляю колонки из grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_remove_columns")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_revision = _normalize_required_str(revision, "revision")

    if not column_slugs:
        raise ToolError("Параметр column_slugs должен содержать хотя бы один слаг.")
    cleaned_slugs = [_normalize_required_str(slug, "column_slugs[*]") for slug in column_slugs]

    http_client = _get_http_client(ctx)
    return await _request(
        method="DELETE",
        path=f"/grids/{normalized_grid_id}/columns",
        body={"revision": normalized_revision, "column_slugs": cleaned_slugs},
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_grid_update_cells(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    revision: Annotated[str, Field(description="Текущая ревизия таблицы")],
    cells: Annotated[list[dict], Field(description="Список ячеек {row_id, column_slug, value}")],
    ctx: Context,
) -> dict:
    """Write: обновить значения ячеек динамической таблицы."""
    await ctx.info(f"Обновляю {len(cells)} ячеек в grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_update_cells")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_revision = _normalize_required_str(revision, "revision")

    if not cells:
        raise ToolError("Параметр cells должен содержать хотя бы одну ячейку.")

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/grids/{normalized_grid_id}/cells",
        body={"revision": normalized_revision, "cells": cells},
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_grid_move_rows(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    revision: Annotated[str, Field(description="Текущая ревизия таблицы")],
    row_id: Annotated[str, Field(description="ID первой переносимой строки")],
    ctx: Context,
    position: int | None = Field(default=None, description="Позиция назначения (0-based)"),
    after_row_id: str | None = Field(default=None, description="ID строки, после которой разместить"),
    rows_count: int | None = Field(default=None, description="Количество подряд идущих строк для переноса"),
) -> dict:
    """Write: переместить одну или несколько строк в динамической таблице."""
    await ctx.info(f"Перемещаю строки в grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_move_rows")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_revision = _normalize_required_str(revision, "revision")
    normalized_row_id = _normalize_required_str(row_id, "row_id")

    body: dict[str, Any] = {
        "revision": normalized_revision,
        "row_id": normalized_row_id,
    }
    if position is not None:
        body["position"] = int(position)
    if after_row_id is not None:
        body["after_row_id"] = _normalize_required_str(after_row_id, "after_row_id")
    if rows_count is not None:
        body["rows_count"] = int(rows_count)

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/grids/{normalized_grid_id}/rows/move",
        body=body,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_grid_move_columns(
    grid_id: Annotated[str, Field(description="ID динамической таблицы")],
    revision: Annotated[str, Field(description="Текущая ревизия таблицы")],
    column_slug: Annotated[str, Field(description="Слаг первой переносимой колонки")],
    ctx: Context,
    position: int | None = Field(default=None, description="Позиция назначения (0-based)"),
    columns_count: int | None = Field(default=None, description="Количество подряд идущих колонок для переноса"),
) -> dict:
    """Write: переместить одну или несколько колонок в динамической таблице."""
    await ctx.info(f"Перемещаю колонки в grid ID={grid_id}")
    _assert_write_enabled("wiki_grid_move_columns")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_revision = _normalize_required_str(revision, "revision")
    normalized_column_slug = _normalize_required_str(column_slug, "column_slug")

    body: dict[str, Any] = {
        "revision": normalized_revision,
        "column_slug": normalized_column_slug,
    }
    if position is not None:
        body["position"] = int(position)
    if columns_count is not None:
        body["columns_count"] = int(columns_count)

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/grids/{normalized_grid_id}/columns/move",
        body=body,
        http_client=http_client,
    )


@mcp.tool(
    tags={"write", "wiki"},
    timeout=60.0,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
async def wiki_grid_clone(
    grid_id: Annotated[str, Field(description="ID исходной динамической таблицы")],
    target: Annotated[str, Field(description="Slug страницы назначения (создаётся, если не существует)")],
    ctx: Context,
    title: str | None = Field(default=None, description="Заголовок таблицы после клонирования (1-255 символов)"),
    with_data: bool = Field(default=False, description="Клонировать вместе с данными (строками)"),
) -> dict:
    """Write: клонировать динамическую таблицу. Асинхронная операция — вернёт task_id для wiki_operation_clone_inline_grid_status."""
    await ctx.info(f"Клонирую grid ID={grid_id} -> {target}")
    _assert_write_enabled("wiki_grid_clone")
    normalized_grid_id = _normalize_grid_id(grid_id)
    normalized_target = _normalize_slug(_normalize_required_str(target, "target"))

    body: dict[str, Any] = {
        "target": normalized_target,
        "with_data": bool(with_data),
    }
    if title is not None:
        stripped_title = title.strip()
        if not stripped_title:
            raise ToolError("Если title передан, он не должен быть пустым.")
        if len(stripped_title) > 255:
            raise ToolError("Параметр title должен быть не длиннее 255 символов.")
        body["title"] = stripped_title

    http_client = _get_http_client(ctx)
    return await _request(
        method="POST",
        path=f"/grids/{normalized_grid_id}/clone",
        body=body,
        http_client=http_client,
    )


def _build_parser(default_transport: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Yandex Wiki MCP server (read/write + readonly mode).",
    )
    parser.add_argument(
        "--transport",
        default=os.getenv("TRANSPORT", default_transport),
        help="MCP transport, например: stdio, http, streamable-http.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "127.0.0.1"),
        help="Host для HTTP транспорта.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8088")),
        help="Port для HTTP транспорта.",
    )
    parser.add_argument(
        "--path",
        default=os.getenv("MCP_PATH", "/mcp"),
        help="Path для HTTP транспорта.",
    )
    parser.add_argument(
        "--readonly",
        action="store_true",
        help="Отключает все write-инструменты (только чтение страниц).",
    )
    return parser


def _configure_readonly(cli_readonly: bool, forced_readonly: bool = False) -> bool:
    return forced_readonly or cli_readonly or _ReadonlyEnv().readonly


def _run_mcp(transport: str, host: str, port: int, path: str):
    run_kwargs = {"transport": transport}
    if transport in HTTP_TRANSPORTS:
        run_kwargs.update({"host": host, "port": port, "path": path})
    mcp.run(**run_kwargs)


def main(
    argv: list[str] | None = None,
    *,
    forced_readonly: bool = False,
    default_transport: str = "stdio",
):
    global SERVER_READONLY

    parser = _build_parser(default_transport=default_transport)
    args = parser.parse_args(argv)
    SERVER_READONLY = _configure_readonly(
        cli_readonly=bool(args.readonly),
        forced_readonly=forced_readonly,
    )
    _run_mcp(
        transport=str(args.transport),
        host=str(args.host),
        port=int(args.port),
        path=str(args.path),
    )


def main_readonly(argv: list[str] | None = None):
    main(argv=argv, forced_readonly=True, default_transport="stdio")
