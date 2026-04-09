"""Tenant router guard.

The heavy lifting (path extraction, lookup, caching, session binding) lives
in :mod:`aegra_api.core.orm` to avoid a circular import. This module just
provides :func:`validate_tenant`, the router-level dependency that rejects
requests for unknown or disabled tenants.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status

from aegra_api.core.orm import Tenant, get_current_tenant


async def validate_tenant(
    tenant: Tenant | None = Depends(get_current_tenant),
) -> Tenant:
    """Raise 404 for unknown tenants, 403 for disabled tenants."""
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )
    if not tenant.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tenant '{tenant.uuid}' is disabled",
        )
    return tenant


__all__ = ["get_current_tenant", "validate_tenant"]
