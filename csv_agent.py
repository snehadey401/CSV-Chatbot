import pandas as pd

# Load CSV once
df = pd.read_csv("data/StudentPerformanceFactors.csv")


def process_calculation(question):
    question = question.lower()

    # Highest score with details
    if ("highest" in question or "best" in question or "top" in question) and ("score" in question or "marks" in question):

        top = df.loc[df["Exam_Score"].idxmax()]

        return f"""
Highest Exam Score : {top['Exam_Score']}

Hours Studied : {top['Hours_Studied']}
Attendance : {top['Attendance']}
Previous Scores : {top['Previous_Scores']}
Gender : {top['Gender']}
Motivation Level : {top['Motivation_Level']}
Family Income : {top['Family_Income']}
Teacher Quality : {top['Teacher_Quality']}
"""

    # Average attendance
    if "attendance" in question and "average" in question:
        return f"Average attendance is {df['Attendance'].mean():.2f}%."

    # Average score
    if "average" in question and "score" in question:
        return f"Average exam score is {df['Exam_Score'].mean():.2f}."

    # Highest score
    if "highest" in question and "score" in question:
        return f"Highest exam score is {df['Exam_Score'].max()}."

    # Lowest score
    if "lowest" in question and "score" in question:
        return f"Lowest exam score is {df['Exam_Score'].min()}."

    # Total students
    if "how many students" in question or "total students" in question:
        return f"There are {len(df)} students."

    # Male students
    if "male" in question and "how many" in question:
        return f"There are {(df['Gender']=='Male').sum()} male students."

    # Female students
    if "female" in question and "how many" in question:
        return f"There are {(df['Gender']=='Female').sum()} female students."

    # Average study hours
    if "average" in question and "hours" in question:
        return f"Average study hours is {df['Hours_Studied'].mean():.2f}."

    return None