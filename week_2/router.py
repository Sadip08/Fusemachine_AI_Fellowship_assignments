from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict
import crud,database,schemas,asyncio, time,logging
from database import get_db

router = APIRouter(prefix="/customers", tags=["customers"])

# Endpoint to list customers with pagination [cite: 253, 256-258]
@router.get("/", response_model=List[schemas.CustomerOut])
def read_customers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_customers(db, skip=skip, limit=limit)

# Endpoint to get a specific customer [cite: 254, 261-266]
@router.get("/{customer_id}", response_model=schemas.CustomerOut)
def read_customer(customer_id: int, db: Session = Depends(get_db)):
    db_customer = crud.get_customer(db, customer_id=customer_id)
    if db_customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return db_customer

router = APIRouter()
logger = logging.getLogger(__name__)

# Dashboard Endpoint: Aggregated Concurrency [cite: 411-412]
@router.get("/overall_counts")
async def read_overall_counts(db: Session = Depends(get_db)):
    start_time = time.time()
    logger.info("Starting concurrent overall_counts tasks") # [cite: 449]

    # Factor VIII: asyncio.gather runs all 8 queries at once [cite: 421, 471]
    results = await asyncio.gather(
        crud.get_customers_count(db),
        crud.get_orders_count(db),
        crud.get_products_count(db),
        crud.get_employees_count(db),
        crud.get_offices_count(db),
        crud.get_payments_count(db),
        crud.get_orderdetails_count(db),
        crud.get_productlines_count(db)
    )

    execution_time = time.time() - start_time
    logger.info(f"Concurrency complete in {execution_time:.4f}s") # [cite: 451, 477]

    # Map results to the required JSON structure [cite: 426-436]
    return {
        "customers": results[0],
        "orders": results[1],
        "products": results[2],
        "employees": results[3],
        "offices": results[4],
        "payments": results[5],
        "orderdetails": results[6],
        "productlines": results[7]
    }