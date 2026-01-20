# api/routes/payout.py
@app.post("/payout/initiate")
async def initiate_payout(amount: float, bank_token: str):
    # Verify ledger has sufficient realized PnL
    if not sufficient_profits(amount):
        raise HTTPException(400, "Insufficient realized profits")
    
    # Create Stripe transfer
    transfer = stripe.Transfer.create(
        amount=int(amount * 100),  # cents
        currency="usd",
        destination=bank_token,
        description="Trading profits payout"
    )
    return {"status": "initiated", "transfer_id": transfer.id}