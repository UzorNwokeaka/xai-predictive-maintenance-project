def recommend_action(rul: float) -> str:
    if rul > 80:
        return "Continue normal operation and routine monitoring."
    if rul > 30:
        return "Schedule inspection and monitor degradation trend."
    return "Immediate maintenance intervention recommended."