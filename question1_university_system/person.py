"""
Base Person class and inheritance hierarchy for University Management System
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Person(ABC):
    """Abstract base class for all persons in the university system"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = ""):
        """
        Initialize a Person with basic information
        
        Args:
            person_id (str): Unique identifier for the person
            name (str): Full name of the person
            email (str): Email address
            phone (str): Phone number (optional)
        """
        self._person_id = person_id
        self._name = name
        self._email = email
        self._phone = phone
    
    # Getter methods
    @property
    def person_id(self) -> str:
        return self._person_id
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def email(self) -> str:
        return self._email
    
    @property
    def phone(self) -> str:
        return self._phone
    
    # Setter methods with validation
    @name.setter
    def name(self, value: str):
        if not value or not value.strip():
            raise ValueError("Name cannot be empty")
        self._name = value.strip()
    
    @email.setter
    def email(self, value: str):
        if "@" not in value or "." not in value:
            raise ValueError("Invalid email format")
        self._email = value
    
    @phone.setter
    def phone(self, value: str):
        self._phone = value
    
    @abstractmethod
    def get_responsibilities(self) -> List[str]:
        """Abstract method to get responsibilities - must be implemented by subclasses"""
        pass
    
    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the person"""
        return {
            'id': self.person_id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'type': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.name} (ID: {self.person_id})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.person_id}', '{self.name}', '{self.email}')"


class Student(Person):
    """Base class for all students"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "", 
                 major: str = "", year: int = 1):
        """
        Initialize a Student
        
        Args:
            person_id (str): Unique student ID
            name (str): Student name
            email (str): Student email
            phone (str): Phone number
            major (str): Student's major
            year (int): Current academic year
        """
        super().__init__(person_id, name, email, phone)
        self._major = major
        self._year = year
        self._enrolled_courses = []  # List of course IDs
        self._grades = {}  # Dictionary: {course_id: grade}
        self._semester_gpas = {}  # Dictionary: {semester: gpa}
    
    @property
    def major(self) -> str:
        return self._major
    
    @major.setter
    def major(self, value: str):
        self._major = value
    
    @property
    def year(self) -> int:
        return self._year
    
    @year.setter
    def year(self, value: int):
        if value < 1 or value > 6:
            raise ValueError("Year must be between 1 and 6")
        self._year = value
    
    def get_responsibilities(self) -> List[str]:
        """Get student responsibilities"""
        return [
            "Attend classes regularly",
            "Complete assignments on time",
            "Maintain academic standards",
            "Follow university policies"
        ]


class Faculty(Person):
    """Base class for all faculty members"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 department: str = "", hire_date: str = ""):
        """
        Initialize a Faculty member
        
        Args:
            person_id (str): Unique faculty ID
            name (str): Faculty name
            email (str): Faculty email
            phone (str): Phone number
            department (str): Department affiliation
            hire_date (str): Date of hire
        """
        super().__init__(person_id, name, email, phone)
        self._department = department
        self._hire_date = hire_date
        self._assigned_courses = []  # List of course IDs
        self._office_hours = {}  # Dictionary: {day: time}
    
    @property
    def department(self) -> str:
        return self._department
    
    @department.setter
    def department(self, value: str):
        self._department = value
    
    @property
    def hire_date(self) -> str:
        return self._hire_date
    
    @abstractmethod
    def calculate_workload(self) -> int:
        """Abstract method to calculate workload - implemented by subclasses"""
        pass
    
    def get_responsibilities(self) -> List[str]:
        """Get general faculty responsibilities"""
        return [
            "Teach assigned courses",
            "Grade assignments and exams",
            "Hold office hours",
            "Participate in faculty meetings"
        ]


class Staff(Person):
    """Class for university staff members"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 department: str = "", position: str = ""):
        """
        Initialize a Staff member
        
        Args:
            person_id (str): Unique staff ID
            name (str): Staff name
            email (str): Staff email
            phone (str): Phone number
            department (str): Department
            position (str): Job position
        """
        super().__init__(person_id, name, email, phone)
        self._department = department
        self._position = position
    
    @property
    def department(self) -> str:
        return self._department
    
    @property
    def position(self) -> str:
        return self._position
    
    def get_responsibilities(self) -> List[str]:
        """Get staff responsibilities"""
        return [
            "Support university operations",
            "Assist students and faculty",
            "Maintain administrative records",
            "Follow departmental procedures"
        ]


# Faculty Subclasses

class Professor(Faculty):
    """Professor class - senior faculty member"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 department: str = "", hire_date: str = "", research_area: str = ""):
        super().__init__(person_id, name, email, phone, department, hire_date)
        self._research_area = research_area
        self._phd_students = []  # List of PhD student IDs
    
    @property
    def research_area(self) -> str:
        return self._research_area
    
    def calculate_workload(self) -> int:
        """Calculate professor workload (courses + research + supervision)"""
        base_load = len(self._assigned_courses) * 3  # 3 hours per course
        research_load = 10  # Fixed research hours
        supervision_load = len(self._phd_students) * 2  # 2 hours per PhD student
        return base_load + research_load + supervision_load
    
    def get_responsibilities(self) -> List[str]:
        """Get professor-specific responsibilities"""
        responsibilities = super().get_responsibilities()
        responsibilities.extend([
            "Conduct research",
            "Supervise PhD students",
            "Publish academic papers",
            "Apply for research grants"
        ])
        return responsibilities


class Lecturer(Faculty):
    """Lecturer class - teaching-focused faculty"""
    
    def calculate_workload(self) -> int:
        """Calculate lecturer workload (primarily teaching)"""
        return len(self._assigned_courses) * 4  # 4 hours per course
    
    def get_responsibilities(self) -> List[str]:
        """Get lecturer-specific responsibilities"""
        responsibilities = super().get_responsibilities()
        responsibilities.extend([
            "Focus on teaching excellence",
            "Develop course materials",
            "Mentor undergraduate students"
        ])
        return responsibilities


class TA(Faculty):
    """Teaching Assistant class"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 department: str = "", hire_date: str = "", supervisor_id: str = ""):
        super().__init__(person_id, name, email, phone, department, hire_date)
        self._supervisor_id = supervisor_id
        self._max_courses = 2  # TAs can only handle 2 courses maximum
    
    @property
    def supervisor_id(self) -> str:
        return self._supervisor_id
    
    def calculate_workload(self) -> int:
        """Calculate TA workload (limited teaching assistance)"""
        return len(self._assigned_courses) * 2  # 2 hours per course
    
    def get_responsibilities(self) -> List[str]:
        """Get TA-specific responsibilities"""
        return [
            "Assist with laboratory sessions",
            "Grade assignments",
            "Hold study sessions",
            "Support course instructor"
        ]


# Student Subclasses

class UndergraduateStudent(Student):
    """Undergraduate student class"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 major: str = "", year: int = 1):
        super().__init__(person_id, name, email, phone, major, year)
        self._max_credits = 18  # Maximum credits per semester
        self._graduation_credits_required = 120
    
    @property
    def max_credits(self) -> int:
        return self._max_credits
    
    def get_responsibilities(self) -> List[str]:
        """Get undergraduate-specific responsibilities"""
        responsibilities = super().get_responsibilities()
        responsibilities.extend([
            "Complete general education requirements",
            "Declare major by sophomore year",
            "Maintain minimum GPA for graduation"
        ])
        return responsibilities


class GraduateStudent(Student):
    """Graduate student class"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 major: str = "", year: int = 1, advisor_id: str = "", 
                 degree_type: str = "Masters"):
        super().__init__(person_id, name, email, phone, major, year)
        self._advisor_id = advisor_id
        self._degree_type = degree_type  # "Masters" or "PhD"
        self._max_credits = 15 if degree_type == "Masters" else 12
        self._thesis_topic = ""
    
    @property
    def advisor_id(self) -> str:
        return self._advisor_id
    
    @property
    def degree_type(self) -> str:
        return self._degree_type
    
    @property
    def thesis_topic(self) -> str:
        return self._thesis_topic
    
    @thesis_topic.setter
    def thesis_topic(self, value: str):
        self._thesis_topic = value
    
    def get_responsibilities(self) -> List[str]:
        """Get graduate-specific responsibilities"""
        responsibilities = super().get_responsibilities()
        responsibilities.extend([
            "Conduct research under advisor supervision",
            "Complete thesis/dissertation",
            "Present research findings",
            "Maintain higher GPA standards"
        ])
        return responsibilities
