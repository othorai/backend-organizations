#routers/oragnization.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from pydantic import BaseModel
from sqlalchemy import text
from app.schemas import schemas
from app.models import models
from app.utils import auth
from app.models.models import Organization, User,user_organizations
from app.utils.auth import get_current_user
from app.utils.database import get_db
from pydantic import BaseModel, EmailStr
from sqlalchemy import select, and_, delete

router = APIRouter()

class SafeUser(BaseModel):
    id: int
    username: str = ""
    email: EmailStr
    role: str
    data_access: str = ""
    is_active: bool = True
    is_admin: bool = False

    class Config:
        from_attributes = True

@router.post("/organizations/", response_model=schemas.Organization)
def create_organization(
    org: schemas.OrganizationCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    try:
        # Create the organization
        db_org = models.Organization(
            name=org.name,
            is_demo=org.is_demo,
            data_source_connected=org.data_source_connected,
            created_by=current_user["user"].id  # Set the creator
        )
        db.add(db_org)
        db.commit()
        db.refresh(db_org)
        
        # Add the creator as admin
        user_org = {
            "user_id": current_user["user"].id,
            "organization_id": db_org.id,
            "role": "admin"
        }
        stmt = models.user_organizations.insert().values(**user_org)
        db.execute(stmt)
        db.commit()
        
        return db_org
        
    except Exception as e:
        db.rollback()
        print(f"Error creating organization: {str(e)}")  # For debugging
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create organization: {str(e)}"
        )

@router.put("/organizations/{org_id}/users/{user_id}/role", response_model=schemas.OrganizationMembershipResponse)
def update_user_role(
    org_id: int,
    user_id: int,
    role_update: schemas.UserRole,
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    # Check if current user is admin in this organization
    if not current_user["user"].is_org_admin(org_id):
        raise HTTPException(
            status_code=403, 
            detail="Only organization admins can update roles"
        )
    
    # Get the target user
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if user is in organization
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org or org not in target_user.organizations:
        raise HTTPException(
            status_code=404, 
            detail="User is not a member of this organization"
        )
    
    # Check if this would remove the last admin
    if role_update.role != 'admin':
        admin_count = db.query(user_organizations).filter(
            user_organizations.c.organization_id == org_id,
            user_organizations.c.role == 'admin'
        ).count()
        if admin_count <= 1 and target_user.is_org_admin(org_id):
            raise HTTPException(
                status_code=400,
                detail="Cannot remove the last admin from organization"
            )
    
    # Update the role
    stmt = user_organizations.update().where(
        user_organizations.c.user_id == user_id,
        user_organizations.c.organization_id == org_id
    ).values(role=role_update.role)
    db.execute(stmt)
    db.commit()
    
    return {
        "message": f"User role updated to {role_update.role}",
        "success": True
    }

@router.get("/organizations/", response_model=List[schemas.Organization])
def list_organizations(
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    try:
        user = current_user["user"]
        orgs = user.organizations
        
        # For debugging
        for org in orgs:
            print(f"Organization {org.name}: created_by={org.created_by}")
        
        return orgs
    except Exception as e:
        print(f"Error listing organizations: {str(e)}")  # For debugging
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list organizations: {str(e)}"
        )

@router.get("/organizations/{org_id}", response_model=schemas.Organization)
def get_organization(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    user = current_user["user"]
    if user.is_admin or org_id in [org.id for org in user.organizations]:
        org = db.query(models.Organization).filter(models.Organization.id == org_id).first()
        if org is None:
            raise HTTPException(status_code=404, detail="Organization not found")
        return org
    raise HTTPException(status_code=403, detail="Not authorized to access this organization")

@router.put("/organizations/{org_id}", response_model=schemas.Organization)
def update_organization(
    org_id: int,
    org_update: schemas.OrganizationUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    # Get the organization
    org = db.query(models.Organization).filter(models.Organization.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Check if user is admin of this specific organization
    user_role = db.query(user_organizations).filter(
        user_organizations.c.user_id == current_user["user"].id,
        user_organizations.c.organization_id == org_id,
        user_organizations.c.role == 'admin'
    ).first()
    
    if not user_role:
        raise HTTPException(status_code=403, detail="Only organization admins can update organizations")
    
    # Update the organization
    for key, value in org_update.dict(exclude_unset=True).items():
        setattr(org, key, value)
    
    db.commit()
    db.refresh(org)
    return org

@router.get("/organizations/{org_id}/users", response_model=List[SafeUser])
def list_users_in_organization(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    try:
        print(f"Checking access for user {current_user['user'].email} in org {org_id}")
        
        # Get organization and check if it exists
        stmt = select(Organization).where(Organization.id == org_id)
        org = db.execute(stmt).scalar_one_or_none()
        
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

        # Get user's role in this organization
        role_stmt = select(user_organizations.c.role).where(
            and_(
                user_organizations.c.user_id == current_user["user"].id,
                user_organizations.c.organization_id == org_id
            )
        )
        user_role = db.execute(role_stmt).scalar_one_or_none()

        if not user_role:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to access this organization's users"
            )

        # Get all users and their roles in this organization
        users_stmt = select(User, user_organizations.c.role).join(
            user_organizations,
            and_(
                User.id == user_organizations.c.user_id,
                user_organizations.c.organization_id == org_id
            )
        )
        results = db.execute(users_stmt).all()

        # Convert to SafeUser objects
        safe_users = []
        for user, role in results:
            safe_user = SafeUser(
                id=user.id,
                username=user.username or "",
                email=user.email,
                role=role,
                data_access=user.data_access or "",
                is_active=user.is_active,
                is_admin=(role == 'admin')
            )
            safe_users.append(safe_user)

        return safe_users

    except Exception as e:
        print(f"Error in list_users_in_organization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while fetching organization users: {str(e)}"
        )

@router.get("/current/users", response_model=List[schemas.User])
def list_users_in_current_organization(
    current_user: dict = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(status_code=400, detail="No current organization set")
    
    org = db.query(models.Organization).filter(models.Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Current organization not found")
    
    return org.users


@router.delete("/organizations/{org_id}", response_model=dict)
def delete_organization(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    try:
        # Check if user is admin of this organization
        user_role = db.query(user_organizations).filter(
            user_organizations.c.user_id == current_user["user"].id,
            user_organizations.c.organization_id == org_id,
            user_organizations.c.role == 'admin'
        ).first()

        if not user_role:
            raise HTTPException(
                status_code=403, 
                detail="Only organization admins can delete organizations"
            )

        # Get the organization
        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

        # Delete related records in correct order
        # 1. Delete articles
        db.execute(
            text("DELETE FROM articles WHERE organization_id = :org_id"),
            {"org_id": org_id}
        )

        # 2. Delete user associations
        db.execute(
            text("DELETE FROM user_organizations WHERE organization_id = :org_id"),
            {"org_id": org_id}
        )

        # 3. Finally delete the organization
        db.delete(org)
        db.commit()
        
        return {"message": "Organization has been deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in delete_organization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete organization"
        )

@router.delete("/organizations/{org_id}/leave", response_model=dict)
def leave_organization(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    try:
        user = current_user["user"]
        org = db.query(Organization).filter(Organization.id == org_id).first()
        
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Check if user is in the organization
        if org not in user.organizations:
            raise HTTPException(status_code=404, detail="User is not a member of this organization")

        # Get the user's role in this organization
        user_role = db.query(user_organizations).filter(
            user_organizations.c.user_id == user.id,
            user_organizations.c.organization_id == org_id
        ).first()

        if not user_role:
            raise HTTPException(status_code=404, detail="User not found in organization")

        # If user is admin, check if they're the last admin
        if user_role.role == 'admin':
            admin_count = db.query(user_organizations).filter(
                user_organizations.c.organization_id == org_id,
                user_organizations.c.role == 'admin'
            ).count()

            if admin_count <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot leave organization as you are the only admin. Please make someone else an admin first."
                )

        # Remove user from organization using direct SQL to bypass trigger
        db.execute(
            f"""
            DELETE FROM user_organizations 
            WHERE user_id = :user_id AND organization_id = :org_id
            """,
            {"user_id": user.id, "org_id": org_id}
        )
        
        db.commit()
        return {"message": "Successfully left the organization"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in leave_organization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to leave organization. Please try again."
        )

@router.get("/organizations/debug", response_model=List[Dict])
async def debug_organizations(
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    if not current_user["user"].is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    orgs = db.query(Organization).all()
    return [{
        "id": org.id,
        "name": org.name,
        "created_by": org.created_by,
        "user_count": len(org.users),
        "admin_count": db.query(user_organizations).filter(
            user_organizations.c.organization_id == org.id,
            user_organizations.c.role == 'admin'
        ).count()
    } for org in orgs]

@router.get("/organizations/{org_id}/role", response_model=Dict[str, str])
def get_user_role_in_organization(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth.get_current_user)
):
    stmt = select(user_organizations.c.role).where(
        and_(
            user_organizations.c.user_id == current_user["user"].id,
            user_organizations.c.organization_id == org_id
        )
    )
    role = db.execute(stmt).scalar_one_or_none()
    return {"role": role or "member"}