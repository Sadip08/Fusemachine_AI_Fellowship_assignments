from pydantic import BaseModel, ConfigDict
from typing import Optional

# This is the "Base" that contains fields common to all versions of a Customer
class CustomerBase(BaseModel):
    customerName: str
    contactLastName: str
    contactFirstName: str
    phone: str
    addressLine1: str
    addressLine2: Optional[str] = None
    city: str
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: str
    salesRepEmployeeNumber: Optional[int] = None
    creditLimit: Optional[float] = None

# Blueprint for creating a new customer 
class CustomerCreate(CustomerBase):
    pass

# Blueprint for updating a customer (all fields optional) 
class CustomerUpdate(BaseModel):
    customerName: Optional[str] = None
    contactLastName: Optional[str] = None
    contactFirstName: Optional[str] = None
    phone: Optional[str] = None

# Blueprint for what the API sends back to the user
class CustomerOut(CustomerBase):
    customerNumber: int # The ID is included in the output [cite: 237]

    # Modern Pydantic V2 Configuration
    model_config = ConfigDict(from_attributes=True)