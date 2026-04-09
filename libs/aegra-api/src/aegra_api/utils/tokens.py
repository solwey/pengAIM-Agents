from functools import lru_cache

from pydantic import BaseModel
from sentry_sdk.integrations import httpx
from jose import JWTError, jwt

class TokenPayload(BaseModel):
    user_id: str
    realm: str
    team_id: str | None
    role: str | None
    claims: dict

@lru_cache(maxsize=16)
def _fetch_jwks(jwks_uri: str) -> dict:
    """Fetch and cache JWKS from the given URI."""
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(jwks_uri)
        resp.raise_for_status()
        return resp.json()


def _find_jwk(jwks: dict, kid: str) -> dict | None:
    return next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)

def _is_keycloak_issuer(domain: str, issuer: str) -> bool:
    """Check whether the issuer belongs to the configured Keycloak domain."""
    kc_base = domain.rstrip("/")
    return issuer.startswith(f"{kc_base}/realms/")

def _extract_group(claims: dict, prefix: str) -> str | None:
    groups = reversed(claims.get("groups", []))
    value = None
    for g in groups:
        name = g.lstrip("/")
        if name.startswith(f"{prefix}:"):
            value = name[len(f"{prefix}:"):]
            break

    return value

def _extract_realm(kc_claims: dict) -> str | None:
    issuer = kc_claims.get("iss") or ""
    if "/realms/" not in issuer:
        return None
    return issuer.rsplit("/realms/", 1)[-1] or None

def decode_keycloak_token(token: str, domain: str) -> TokenPayload:
    """Validate a Keycloak access token using the realm's JWKS.

    Supports any realm/client on the configured KEYCLOAK_URL domain.
    The realm is extracted from the token's ``iss`` claim.
    """
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError:
        raise ValueError("Invalid token: malformed header")

    kid = unverified_header.get("kid")
    if not kid:
        raise ValueError("Invalid token: missing kid")

    try:
        unverified_claims = jwt.get_unverified_claims(token)
    except JWTError:
        raise ValueError("Invalid token: malformed payload")

    issuer = unverified_claims.get("iss", "")

    if not _is_keycloak_issuer(domain, issuer):
        raise ValueError("Invalid token: issuer not recognised")

    jwks_uri = f"{issuer}/protocol/openid-connect/certs"

    jwks = _fetch_jwks(jwks_uri)
    key = _find_jwk(jwks, kid)

    if not key:
        # JWKS may be stale after key rotation - clear cache and retry once
        _fetch_jwks.cache_clear()
        jwks = _fetch_jwks(jwks_uri)
        key = _find_jwk(jwks, kid)
        if not key:
            raise ValueError("Invalid token: unknown signing key")

    algorithm = key.get("alg", unverified_header.get("alg", "RS256"))

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[algorithm],
            issuer=issuer,
            options={"verify_aud": False, "verify_iss": True},
        )
    except JWTError as e:
        raise ValueError(f"Invalid token: {e!s}")

    user_id = claims.get("sub")
    if not user_id:
        raise ValueError("Invalid token: Missing sub claim")

    realm = _extract_realm(claims)
    if not realm:
        raise ValueError("Invalid token: Missing realm claim")

    team_id = _extract_group(claims, prefix="team")
    role = _extract_group(claims, prefix="role")

    return TokenPayload(
        user_id=str(user_id),
        team_id=team_id,
        role=role,
        realm=realm,
        claims=claims
    )
