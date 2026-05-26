def add_engagement(X):
    X = X.copy()
    study_hours = X["weekly_self_study_hours"].astype(float)
    attendance = X["attendance_percentage"].astype(float)
    participation = X["class_participation"].astype(float)

    normalized_study = study_hours.clip(0, 30) / 30
    normalized_attendance = attendance.clip(0, 100) / 100
    normalized_participation = participation.clip(0, 10) / 10

    X["engagement"] = (
        normalized_study + normalized_attendance + normalized_participation
    ) / 3 * 100
    X["study_attendance_balance"] = normalized_study * normalized_attendance * 100
    X["participation_intensity"] = normalized_participation * (1 + normalized_study) * 50
    X["consistency_index"] = (
        0.45 * normalized_attendance
        + 0.35 * normalized_study
        + 0.20 * normalized_participation
    ) * 100
    X["support_need_index"] = 100 - X["consistency_index"]

    return X
