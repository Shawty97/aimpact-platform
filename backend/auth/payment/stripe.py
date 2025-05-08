"""
Stripe Integration Module for Subscription Management

This module handles Stripe integration for subscription management, including:
1. Webhook handlers for subscription lifecycle events
2. Checkout session creation
3. Subscription management API
4. Plan management
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.auth.db.database import get_db
from backend.auth.db.models import Tenant, User, Plan, Subscription
from backend.auth.api.dependencies import get_current_active_user

# Configure logging
logger = logging.getLogger("backend.auth.payment.stripe")

# Initialize Stripe
stripe.api_key = settings.STRIPE_API_KEY
stripe.api_version = "2023-10-16"  # Use latest stable API version

# Router for payment endpoints
router = APIRouter()

# ---------------------- Helper Functions ----------------------

def get_stripe_customer_id(tenant_id: str, db: Session) -> Optional[str]:
    """
    Get the Stripe customer ID for a tenant, or create one if it doesn't exist.
    """
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        return None
        
    # Check if tenant has settings with Stripe customer ID
    settings_dict = tenant.settings or {}
    
    if "stripe_customer_id" in settings_dict:
        return settings_dict["stripe_customer_id"]
        
    # No Stripe customer ID found, so create one
    try:
        customer = stripe.Customer.create(
            name=tenant.name,
            metadata={"tenant_id": tenant.id}
        )
        
        # Save customer ID to tenant settings
        settings_dict["stripe_customer_id"] = customer.id
        tenant.settings = settings_dict
        db.commit()
        
        return customer.id
    except stripe.error.StripeError as e:
        logger.error(f"Error creating Stripe customer: {str(e)}")
        return None


def create_or_update_subscription(
    tenant_id: str,
    plan_id: str,
    stripe_subscription_id: str,
    status: str,
    current_period_start: datetime,
    current_period_end: datetime,
    db: Session
) -> Subscription:
    """
    Create or update a subscription in the database.
    """
    # Check if subscription already exists
    subscription = db.query(Subscription).filter(
        Subscription.stripe_subscription_id == stripe_subscription_id
    ).first()
    
    if subscription:
        # Update existing subscription
        subscription.plan_id = plan_id
        subscription.status = status
        subscription.current_period_start = current_period_start
        subscription.current_period_end = current_period_end
        subscription.updated_at = datetime.utcnow()
    else:
        # Create new subscription
        subscription = Subscription(
            tenant_id=tenant_id,
            plan_id=plan_id,
            stripe_subscription_id=stripe_subscription_id,
            status=status,
            current_period_start=current_period_start,
            current_period_end=current_period_end
        )
        db.add(subscription)
        
    db.commit()
    db.refresh(subscription)
    return subscription


def find_plan_by_stripe_price_id(stripe_price_id: str, db: Session) -> Optional[Plan]:
    """
    Find a plan by its Stripe price ID.
    """
    return db.query(Plan).filter(Plan.stripe_price_id == stripe_price_id).first()


def convert_stripe_timestamp(timestamp: int) -> datetime:
    """
    Convert a Stripe timestamp to a datetime object.
    """
    return datetime.fromtimestamp(timestamp)


# ---------------------- Webhook Handlers ----------------------

async def handle_subscription_created(event: Dict[str, Any], db: Session) -> None:
    """
    Handle subscription.created event from Stripe.
    """
    subscription = event["data"]["object"]
    tenant_id = subscription["metadata"].get("tenant_id")
    
    if not tenant_id:
        logger.error("Subscription created without tenant_id in metadata")
        return
        
    # Get the plan from the price ID
    items = subscription["items"]["data"]
    if not items:
        logger.error("Subscription created without items")
        return
        
    price_id = items[0]["price"]["id"]
    plan = find_plan_by_stripe_price_id(price_id, db)
    
    if not plan:
        logger.error(f"No plan found for price ID: {price_id}")
        return
        
    # Create subscription record
    create_or_update_subscription(
        tenant_id=tenant_id,
        plan_id=plan.id,
        stripe_subscription_id=subscription["id"],
        status=subscription["status"],
        current_period_start=convert_stripe_timestamp(subscription["current_period_start"]),
        current_period_end=convert_stripe_timestamp(subscription["current_period_end"]),
        db=db
    )
    
    logger.info(f"Created subscription for tenant {tenant_id}")


async def handle_subscription_updated(event: Dict[str, Any], db: Session) -> None:
    """
    Handle subscription.updated event from Stripe.
    """
    subscription = event["data"]["object"]
    
    # Find existing subscription
    db_subscription = db.query(Subscription).filter(
        Subscription.stripe_subscription_id == subscription["id"]
    ).first()
    
    if not db_subscription:
        logger.error(f"Subscription not found: {subscription['id']}")
        return
        
    # Check if plan changed
    items = subscription["items"]["data"]
    if items:
        price_id = items[0]["price"]["id"]
        plan = find_plan_by_stripe_price_id(price_id, db)
        
        if plan and plan.id != db_subscription.plan_id:
            db_subscription.plan_id = plan.id
    
    # Update subscription status and dates
    db_subscription.status = subscription["status"]
    db_subscription.current_period_start = convert_stripe_timestamp(subscription["current_period_start"])
    db_subscription.current_period_end = convert_stripe_timestamp(subscription["current_period_end"])
    db_subscription.updated_at = datetime.utcnow()
    
    db.commit()
    logger.info(f"Updated subscription {subscription['id']}")


async def handle_subscription_deleted(event: Dict[str, Any], db: Session) -> None:
    """
    Handle subscription.deleted event from Stripe.
    """
    subscription = event["data"]["object"]
    
    # Find and update subscription
    db_subscription = db.query(Subscription).filter(
        Subscription.stripe_subscription_id == subscription["id"]
    ).first()
    
    if not db_subscription:
        logger.error(f"Subscription not found: {subscription['id']}")
        return
        
    # Mark subscription as canceled
    db_subscription.status = "canceled"
    db_subscription.updated_at = datetime.utcnow()
    
    db.commit()
    logger.info(f"Canceled subscription {subscription['id']}")


async def handle_checkout_session_completed(event: Dict[str, Any], db: Session) -> None:
    """
    Handle checkout.session.completed event from Stripe.
    """
    session = event["data"]["object"]
    
    # Process only subscription checkouts
    if session["mode"] != "subscription":
        return
        
    # Get tenant from metadata
    tenant_id = session.get("client_reference_id") or session["metadata"].get("tenant_id")
    
    if not tenant_id:
        logger.error("Checkout completed without tenant reference")
        return
        
    # The subscription ID will be available here
    subscription_id = session["subscription"]
    
    try:
        # Fetch full subscription object from Stripe
        subscription = stripe.Subscription.retrieve(subscription_id)
        
        # Get the plan from the price ID
        items = subscription["items"]["data"]
        if not items:
            logger.error("Subscription created without items")
            return
            
        price_id = items[0]["price"]["id"]
        plan = find_plan_by_stripe_price_id(price_id, db)
        
        if not plan:
            logger.error(f"No plan found for price ID: {price_id}")
            return
            
        # Create or update subscription record
        create_or_update_subscription(
            tenant_id=tenant_id,
            plan_id=plan.id,
            stripe_subscription_id=subscription_id,
            status=subscription["status"],
            current_period_start=convert_stripe_timestamp(subscription["current_period_start"]),
            current_period_end=convert_stripe_timestamp(subscription["current_period_end"]),
            db=db
        )
        
        logger.info(f"Processed checkout session for tenant {tenant_id}")
        
    except stripe.error.StripeError as e:
        logger.error(f"Error processing checkout session: {str(e)}")


async def handle_payment_failed(event: Dict[str, Any], db: Session) -> None:
    """
    Handle invoice.payment_failed event from Stripe.
    """
    invoice = event["data"]["object"]
    subscription_id = invoice.get("subscription")
    
    if not subscription_id:
        return
        
    # Find subscription
    db_subscription = db.query(Subscription).filter(
        Subscription.stripe_subscription_id == subscription_id
    ).first()
    
    if not db_subscription:
        logger.error(f"Subscription not found: {subscription_id}")
        return
        
    # Update subscription status
    db_subscription.status = "past_due"
    db_subscription.updated_at = datetime.utcnow()
    
    db.commit()
    logger.info(f"Marked subscription {subscription_id} as past_due due to payment failure")


# Map of event types to handler functions
EVENT_HANDLERS = {
    "checkout.session.completed": handle_checkout_session_completed,
    "subscription.created": handle_subscription_created,
    "subscription.updated": handle_subscription_updated,
    "subscription.deleted": handle_subscription_deleted,
    "invoice.payment_failed": handle_payment_failed
}


# ---------------------- API Routes ----------------------

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Webhook endpoint for Stripe events.
    """
    # Get the signature from the header
    signature = request.headers.get("stripe-signature")
    
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature header")
        
    try:
        # Get raw body
        body = await request.body()
        
        # Verify signature and construct event
        event = stripe.Webhook.construct_event(
            payload=body,
            sig_header=signature,
            secret=settings.STRIPE_WEBHOOK_SECRET
        )
        
        # Get event type and handle it
        event_type = event["type"]
        handler = EVENT_HANDLERS.get(event_type)
        
        if handler:
            # Process event in the background
            background_tasks.add_task(handler, event, db)
            logger.info(f"Processing Stripe webhook event: {event_type}")
        else:
            logger.info(f"Unhandled Stripe webhook event type: {event_type}")
            
        return {"success": True}
        
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid Stripe webhook signature")
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")
    except Exception as e:
        logger.error(f"Error processing Stripe webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans")
async def list_plans(db: Session = Depends(get_db)):
    """
    List all available subscription plans.
    """
    plans = db.query(Plan).all()
    return plans


@router.post("/checkout-session")
async def create_checkout_session(
    plan_id: str,
    success_url: str,
    cancel_url: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a Stripe checkout session for subscription.
    """
    # Get the plan
    plan = db.query(Plan).filter(Plan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
        
    # Get the tenant ID
    tenant_id = getattr(current_user, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant associated with user")
        
    # Get or create Stripe customer
    customer_id = get_stripe_customer_id(tenant_id, db)
    if not customer_id:
        raise HTTPException(status_code=500, detail="Failed to create Stripe customer")
        
    try:
        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[
                {
                    "price": plan.stripe_price_id,
                    "quantity": 1,
                },
            ],
            mode="subscription",
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=tenant_id,
            metadata={
                "tenant_id": tenant_id,
                "plan_id": plan_id,
            },
        )
        
        return {"checkout_url": checkout_session.url}
        
    except stripe.error.StripeError as e:
        logger.error(f"Error creating checkout session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subscription")
async def get_subscription(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the current subscription for the user's tenant.
    """
    tenant_id = getattr(current_user, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant associated with user")
        
    # Get active subscription
    subscription = db.query(Subscription).filter(
        Subscription.tenant_id == tenant_id,
        Subscription.status.in_(["active", "trialing", "past_due"])
    ).order_by(Subscription.current_period_end.desc()).first()
    
    if not subscription:
        return {"subscription": None, "plan": None}
        
    # Get plan details
    plan = db.query(Plan).filter(Plan.id == subscription.plan_id).first()
    
    return {
        "subscription": subscription,
        "plan": plan
    }


@router.post("/subscription/cancel")
async def cancel_subscription(
    cancel_immediately: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Cancel the current subscription for the user's tenant.
    
    Args:
        cancel_immediately: If True, cancel immediately. If False (default),
                            cancel at the end of the current billing period.
    """
    tenant_id = getattr(current_user, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant associated with user")
        
    # Get active subscription
    subscription = db.query(Subscription).filter(
        Subscription.tenant_id == tenant_id,
        Subscription.status.in_(["active", "trialing", "past_due"])
    ).order_by(Subscription.current_period_end.desc()).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="No active subscription found")
        
    try:
        # Cancel the subscription in Stripe
        stripe_subscription = stripe.Subscription.modify(
            subscription.stripe_subscription_id,
            cancel_at_period_end=not cancel_immediately,
        )
        
        if cancel_immediately:
            # Immediately cancel the subscription
            stripe_subscription = stripe.Subscription.delete(
                subscription.stripe_subscription_id
            )
            
            # Update local record
            subscription.status = "canceled"
            subscription.updated_at = datetime.utcnow()
            db.commit()
            
            return {
                "message": "Subscription canceled immediately",
                "subscription_id": subscription.id
            }
        else:
            # Update local record to reflect the scheduled cancellation
            subscription.status = "active"  # Remains active until period end
            subscription.updated_at = datetime.utcnow()
            db.commit()
            
            return {
                "message": "Subscription will be canceled at the end of the billing period",
                "subscription_id": subscription.id,
                "end_date": subscription.current_period_end
            }
            
    except stripe.error.StripeError as e:
        logger.error(f"Error canceling subscription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subscription/reactivate")
async def reactivate_subscription(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Reactivate a subscription that was scheduled for cancellation at period end.
    """
    tenant_id = getattr(current_user, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant associated with user")
        
    # Get subscription scheduled for cancellation
    subscription = db.query(Subscription).filter(
        Subscription.tenant_id == tenant_id,
        Subscription.status == "active"
    ).order_by(Subscription.current_period_end.desc()).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="No subscription found that can be reactivated")
        
    try:
        # Check if the subscription is actually scheduled for cancellation
        stripe_subscription = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
        
        if not stripe_subscription.cancel_at_period_end:
            return {
                "message": "Subscription is not scheduled for cancellation",
                "subscription_id": subscription.id
            }
            
        # Remove the cancellation schedule
        stripe_subscription = stripe.Subscription.modify(
            subscription.stripe_subscription_id,
            cancel_at_period_end=False,
        )
        
        # Update local record
        subscription.updated_at = datetime.utcnow()
        db.commit()
        
        return {
            "message": "Subscription reactivated successfully",
            "subscription_id": subscription.id
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Error reactivating subscription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subscription/update")
async def update_subscription(
    plan_id: str,
    proration_date: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update a subscription to a different plan.
    
    Args:
        plan_id: The ID of the new plan
        proration_date: Optional timestamp for proration (default: now)
    """
    tenant_id = getattr(current_user, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant associated with user")
        
    # Get active subscription
    subscription = db.query(Subscription).filter(
        Subscription.tenant_id == tenant_id,
        Subscription.status.in_(["active", "trialing"])
    ).order_by(Subscription.current_period_end.desc()).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="No active subscription found")
        
    # Get the new plan
    plan = db.query(Plan).filter(Plan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
        
    try:
        # Update the subscription in Stripe
        items = [{
            'id': stripe.Subscription.retrieve(subscription.stripe_subscription_id)['items']['data'][0].id,
            'price': plan.stripe_price_id,
        }]
        
        params = {
            'items': items,
            'proration_behavior': 'create_prorations',
        }
        
        if proration_date:
            params['proration_date'] = proration_date
            
        stripe_subscription = stripe.Subscription.modify(
            subscription.stripe_subscription_id,
            **params
        )
        
        # Update local record
        subscription.plan_id = plan_id
        subscription.updated_at = datetime.utcnow()
        db.commit()
        
        return {
            "message": "Subscription updated successfully",
            "subscription_id": subscription.id,
            "new_plan_id": plan_id
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Error updating subscription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

