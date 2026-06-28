def classify_health_status(rul: float) -> str:
    if rul > 80:
        return "Healthy"
    if rul > 30:
        return "Warning"
    return "Critical"


def classify_risk_level(rul: float) -> str:
    if rul > 80:
        return "Low Risk"
    if rul > 30:
        return "Medium Risk"
    return "High Risk"