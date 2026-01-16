from pydantic import BaseModel, Field, validator

class ChurnInput(BaseModel):
    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Edad del cliente (18-100 años)"
    )
    gender: str = Field(
        ...,
        description="Género del cliente (Male/Female)"
    )
    subscription_type: str = Field(
        ...,
        description="Tipo de suscripción (Basic/Standard/Premium)"
    )
    watch_hours: float = Field(
        ...,
        ge=0,
        description="Horas de visualización (debe ser positivo)"
    )
    region: str = Field(
        ...,
        description="Región geográfica del cliente"
    )
    number_of_profiles: int = Field(
        ...,
        ge=1,
        description="Número de perfiles (mínimo 1)"
    )
    payment_method: str = Field(
        ...,
        description="Método de pago"
    )
    device: str = Field(
        ...,
        description="Dispositivo usado para ver contenido"
    )

    # Validador que convierte automáticamente a minúsculas
    @validator('gender', 'subscription_type', 'region', 'payment_method', 'device', pre=True)
    def lowercase_strings(cls, v):
        """Convierte todos los strings a minúsculas"""
        if isinstance(v, str):
            return v.lower().strip()
        return v

    # Validador para edad
    @validator('age')
    def validate_age(cls, v):
        """Valida que la edad sea razonable"""
        if v < 18:
            raise ValueError('La edad debe ser mayor o igual a 18')
        if v > 100:
            raise ValueError('La edad debe ser menor o igual a 100')
        return v

    # Validador para watch_hours
    @validator('watch_hours')
    def validate_watch_hours(cls, v):
        """Valida que las horas sean positivas"""
        if v < 0:
            raise ValueError('Las horas de visualización no pueden ser negativas')
        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "gender": "Male",
                "subscription_type": "Basic",
                "watch_hours": 120.5,
                "region": "Asia",
                "number_of_profiles": 2,
                "payment_method": "debit card",
                "device": "tablet"
            }
        }