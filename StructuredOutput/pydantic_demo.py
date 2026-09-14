from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: int
    email: Optional[EmailStr] = None
    grade: int = Field(ge=1, le=10, description='Student Grade')
    address: str = Field(default="Kathmandu")


new_student = {
    "name": "Manish",
    "age": 22,
    "email": "manish@gmail.com",
    "grade": 9,
    "address": "Kathmandu"
    }

student = Student(**new_student)

student_dict = student.model_dump() #dump object to dict

print(student_dict)

student_json = student.model_dump_json() #dump object to json

print(student_json)