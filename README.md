# Group6COMP258ProjectBackEnd
###
run the backend
###

$env:FLASK_APP = "main"
flask run

###
default port is 5000
api : http://localhost:5000/predict
body:
{
    "INTAKE COLLEGE EXPERIENCE": "New to College",
    "SCHOOL CODE": "CH",
    "STUDENT LEVEL NAME": "Post Secondary",
    "TIME STATUS NAME": "Full-Time",
    "RESIDENCY STATUS NAME": "Resident",
    "FUNDING SOURCE NAME": "GPOG - FT",
    "GENDER": "F",
    "DISABILITY IND": "N",
    "ACADEMIC PERFORMANCE": "C - Satisfactory",
    "FIRST YEAR PERSISTENCE COUNT": "1",
    "ENGLISH TEST SCORE": "Average",
    "AGE GROUP LONG NAME": "19 to 20",
    "APPL FIRST LANGUAGE DESC": "English",
    "APPLICANT CATEGORY NAME": "Post Secondary",
    "APPLICANT TARGET SEGMENT NAME": "Non-Direct Entry",
    "LOCATION": "M",
    "PROGRAM LENGTH": "20",
    "REST SEMESTERS": "0"
}
###