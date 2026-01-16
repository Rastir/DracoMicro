from pydantic import BaseModel

class ChurnInput(BaseModel):
    age: int
    gender: str
    subscription_type: str
    watch_hours: float
    last_login_days: int
    region: str
