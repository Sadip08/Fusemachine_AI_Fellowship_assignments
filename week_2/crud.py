from sqlalchemy.orm import Session
import models
import schemas
import logging

# Initialize logger to track database activity
logger = logging.getLogger(__name__)

# --- Standard Customer CRUD ---

def get_customer(db: Session, customer_id: int):
    """Fetch a single customer by ID."""
    return db.query(models.Customer).filter(models.Customer.customerNumber == customer_id).first()

def get_customers(db: Session, skip: int = 0, limit: int = 100):
    """Fetch a list of customers with pagination."""
    return db.query(models.Customer).offset(skip).limit(limit).all()

def create_customer(db: Session, customer: schemas.CustomerCreate):
    """Create a new customer record."""
    db_customer = models.Customer(**customer.model_dump())
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    return db_customer

# --- Factor VIII: Concurrent Count Functions ---
# These are 'async' so they can be run in parallel by asyncio.gather()

async def get_customers_count(db: Session):
    logger.info("Database: Counting Customers")
    return db.query(models.Customer).count()

async def get_orders_count(db: Session):
    logger.info("Database: Counting Orders")
    return db.query(models.Order).count()

async def get_products_count(db: Session):
    logger.info("Database: Counting Products")
    return db.query(models.Product).count()

async def get_employees_count(db: Session):
    logger.info("Database: Counting Employees")
    return db.query(models.Employee).count()

async def get_offices_count(db: Session):
    logger.info("Database: Counting Offices")
    return db.query(models.Office).count()

async def get_payments_count(db: Session):
    logger.info("Database: Counting Payments")
    return db.query(models.Payment).count()

async def get_orderdetails_count(db: Session):
    logger.info("Database: Counting Order Details")
    return db.query(models.OrderDetail).count()

async def get_productlines_count(db: Session):
    logger.info("Database: Counting Product Lines")
    return db.query(models.ProductLine).count()