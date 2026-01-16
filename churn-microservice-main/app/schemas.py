from pydantic import BaseModel

class ChurnInput(BaseModel):
    age: int
    gender: str
    subscription_type: str
    watch_hours: float
    region: str
    number_of_profiles: int
    payment_method: str
    device: str
