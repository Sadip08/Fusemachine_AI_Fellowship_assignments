from fastapi import FastAPI
import models
from database import engine
from router import router as customer_router

# Create the database tables (if they don't exist)
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Customer API",
    description="An API to manage and view customer data using a 4-layer architecture."
)

# Include the customer router we created in the previous step
app.include_router(customer_router)

@app.get("/")
def root():
    return {"message": "Welcome to the Customer API! Head to /docs for the interactive UI."}