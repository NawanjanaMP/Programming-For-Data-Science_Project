"""
Enhanced Student Management System with course enrollment, GPA calculation, and secure records
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
import statistics

class EnhancedStudent:
    """Enhanced Student class with advanced management capabilities"""
    
    def __init__(self, student_id: str, name: str, email: str, major: str = ""):
        self.student_id = student_id
        self.name = name
        self.email = email
        self.major = major
        self._enrolled_courses = {}  # {course_id: {'semester': str, 'credits': int}}
        self._grades = {}  # {course_id: {'grade': float, 'semester': str, 'credits': int}}
        self._semester_gpas = {}  # {semester: gpa}
        self._overall_gpa = 0.0
        self._total_credits = 0
        self._academic_status = "Good Standing"
    
    def enroll_course(self, course_id: str, semester: str, credits: int = 3) -> bool:
        """
        Enroll student in a course
        
        Args:
            course_id (str): Course identifier
            semester (str): Semester (e.g., "Fall 2024")
            credits (int): Course credit hours
            
        Returns:
            bool: True if enrollment successful, False otherwise
        """
        try:
            # Check if already enrolled
            if course_id in self._enrolled_courses:
                print(f"Already enrolled in {course_id}")
                return False
            
            # Check credit limit (assume 18 credits max per semester)
            current_semester_credits = sum(
                course['credits'] for course in self._enrolled_courses.values()
                if course['semester'] == semester
            )
            
            if current_semester_credits + credits > 18:
                print(f"Credit limit exceeded for {semester}")
                return False
            
            # Enroll in course
            self._enrolled_courses[course_id] = {
                'semester': semester,
                'credits': credits
            }
            
            print(f"Successfully enrolled in {course_id} for {semester}")
            return True
            
        except Exception as e:
            print(f"Enrollment error: {e}")
            return False
    
    def drop_course(self, course_id: str) -> bool:
        """
        Drop a course
        
        Args:
            course_id (str): Course identifier
            
        Returns:
            bool: True if drop successful, False otherwise
        """
        try:
            if course_id not in self._enrolled_courses:
                print(f"Not enrolled in {course_id}")
                return False
            
            # Remove from enrolled courses
            del self._enrolled_courses[course_id]
            
            # Remove grade if exists
            if course_id in self._grades:
                del self._grades[course_id]
            
            print(f"Successfully dropped {course_id}")
            return True
            
        except Exception as e:
            print(f"Drop course error: {e}")
            return False
    
    def add_grade(self, course_id: str, grade: float) -> bool:
        """
        Add grade for a completed course
        
        Args:
            course_id (str): Course identifier
            grade (float): Grade (0.0 - 4.0 scale)
            
        Returns:
            bool: True if grade added successfully
        """
        try:
            # Validate grade
            if not (0.0 <= grade <= 4.0):
                raise ValueError("Grade must be between 0.0 and 4.0")
            
            # Check if student was enrolled in course
            if course_id not in self._enrolled_courses:
                print(f"Student was not enrolled in {course_id}")
                return False
            
            course_info = self._enrolled_courses[course_id]
            
            # Add grade
            self._grades[course_id] = {
                'grade': grade,
                'semester': course_info['semester'],
                'credits': course_info['credits']
            }
            
            # Recalculate GPA
            self._calculate_all_gpas()
            
            print(f"Grade {grade} added for {course_id}")
            return True
            
        except Exception as e:
            print(f"Add grade error: {e}")
            return False
    
    def _calculate_all_gpas(self):
        """Recalculate semester and overall GPAs"""
        # Group grades by semester
        semester_grades = {}
        for course_id, grade_info in self._grades.items():
            semester = grade_info['semester']
            if semester not in semester_grades:
                semester_grades[semester] = []
            semester_grades[semester].append(grade_info)
        
        # Calculate semester GPAs
        for semester, grades in semester_grades.items():
            total_points = sum(g['grade'] * g['credits'] for g in grades)
            total_credits = sum(g['credits'] for g in grades)
            
            if total_credits > 0:
                self._semester_gpas[semester] = total_points / total_credits
        
        # Calculate overall GPA
        if self._grades:
            total_points = sum(g['grade'] * g['credits'] for g in self._grades.values())
            self._total_credits = sum(g['credits'] for g in self._grades.values())
            self._overall_gpa = total_points / self._total_credits if self._total_credits > 0 else 0.0
        
        # Update academic status
        self._update_academic_status()
    
    def calculate_gpa(self, semester: str = None) -> float:
        """
        Calculate GPA for specific semester or overall
        
        Args:
            semester (str): Specific semester, or None for overall GPA
            
        Returns:
            float: GPA value
        """
        if semester:
            return self._semester_gpas.get(semester, 0.0)
        else:
            return self._overall_gpa
    
    def _update_academic_status(self):
        """Update academic status based on GPA"""
        if self._overall_gpa >= 3.5:
            self._academic_status = "Dean's List"
        elif self._overall_gpa >= 2.0:
            self._academic_status = "Good Standing"
        else:
            self._academic_status = "Probation"
    
    def get_academic_status(self) -> str:
        """
        Get current academic status
        
        Returns:
            str: Academic status
        """
        return self._academic_status
    
    def get_enrolled_courses(self) -> Dict[str, Dict]:
        """Get list of currently enrolled courses"""
        return self._enrolled_courses.copy()
    
    def get_transcript(self) -> Dict:
        """Get complete academic transcript"""
        return {
            'student_id': self.student_id,
            'name': self.name,
            'major': self.major,
            'overall_gpa': self._overall_gpa,
            'total_credits': self._total_credits,
            'academic_status': self._academic_status,
            'semester_gpas': self._semester_gpas.copy(),
            'grades': self._grades.copy()
        }


class SecureStudentRecord:
    """Secure student record class with encapsulation and validation"""
    
    def __init__(self, student_id: str, name: str, email: str):
        """
        Initialize secure student record
        
        Args:
            student_id (str): Student ID
            name (str): Student name  
            email (str): Student email
        """
        # Private attributes
        self.__student_id = student_id
        self.__name = ""
        self.__email = ""
        self.__gpa = 0.0
        self.__enrolled_courses = []
        self.__max_enrollment = 6  # Maximum courses per semester
        self.__is_active = True
        
        # Use setters for validation
        self.name = name
        self.email = email
    
    # Getter methods (read-only properties)
    @property
    def student_id(self) -> str:
        """Get student ID (read-only)"""
        return self.__student_id
    
    @property 
    def name(self) -> str:
        """Get student name"""
        return self.__name
    
    @property
    def email(self) -> str:
        """Get student email"""
        return self.__email
    
    @property
    def gpa(self) -> float:
        """Get current GPA"""
        return self.__gpa
    
    @property
    def enrolled_courses(self) -> List[str]:
        """Get enrolled courses (read-only copy)"""
        return self.__enrolled_courses.copy()
    
    @property
    def is_active(self) -> bool:
        """Check if student record is active"""
        return self.__is_active
    
    # Setter methods with validation
    @name.setter
    def name(self, value: str):
        """
        Set student name with validation
        
        Args:
            value (str): New name
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Name must be a non-empty string")
        
        if len(value) < 2:
            raise ValueError("Name must be at least 2 characters long")
        
        if not all(c.isalpha() or c.isspace() for c in value):
            raise ValueError("Name can only contain letters and spaces")
        
        self.__name = value.strip()
    
    @email.setter  
    def email(self, value: str):
        """
        Set email with validation
        
        Args:
            value (str): New email
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Email must be a non-empty string")
        
        if "@" not in value or "." not in value.split("@")[-1]:
            raise ValueError("Invalid email format")
        
        # Basic email validation
        parts = value.split("@")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Invalid email format")
        
        self.__email = value.strip().lower()
    
    def set_gpa(self, value: float):
        """
        Set GPA with validation
        
        Args:
            value (float): New GPA
        """
        if not isinstance(value, (int, float)):
            raise ValueError("GPA must be a number")
        
        if not (0.0 <= value <= 4.0):
            raise ValueError("GPA must be between 0.0 and 4.0")
        
        self.__gpa = float(value)
    
    def enroll_in_course(self, course_id: str) -> bool:
        """
        Enroll in a course with validation
        
        Args:
            course_id (str): Course identifier
            
        Returns:
            bool: True if enrollment successful
        """
        try:
            # Validate inputs
            if not isinstance(course_id, str) or not course_id.strip():
                raise ValueError("Course ID must be a non-empty string")
            
            course_id = course_id.strip().upper()
            
            # Check if already enrolled
            if course_id in self.__enrolled_courses:
                raise ValueError(f"Already enrolled in {course_id}")
            
            # Check enrollment limit
            if len(self.__enrolled_courses) >= self.__max_enrollment:
                raise ValueError(f"Maximum enrollment limit ({self.__max_enrollment}) reached")
            
            # Check if student is active
            if not self.__is_active:
                raise ValueError("Student record is inactive")
            
            # Add to enrolled courses
            self.__enrolled_courses.append(course_id)
            return True
            
        except ValueError as e:
            print(f"Enrollment failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during enrollment: {e}")
            return False
    
    def drop_course(self, course_id: str) -> bool:
        """
        Drop a course
        
        Args:
            course_id (str): Course identifier
            
        Returns:
            bool: True if drop successful
        """
        try:
            if not isinstance(course_id, str) or not course_id.strip():
                raise ValueError("Course ID must be a non-empty string")
            
            course_id = course_id.strip().upper()
            
            if course_id not in self.__enrolled_courses:
                raise ValueError(f"Not enrolled in {course_id}")
            
            self.__enrolled_courses.remove(course_id)
            return True
            
        except ValueError as e:
            print(f"Drop failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during drop: {e}")
            return False
    
    def deactivate_record(self):
        """Deactivate student record"""
        self.__is_active = False
        self.__enrolled_courses.clear()
    
    def reactivate_record(self):
        """Reactivate student record"""
        self.__is_active = True
    
    def get_enrollment_count(self) -> int:
        """Get current number of enrolled courses"""
        return len(self.__enrolled_courses)
    
    def get_secure_info(self) -> Dict[str, Union[str, float, int, bool]]:
        """
        Get secure student information (no direct access to private data)
        
        Returns:
            dict: Safe student information
        """
        return {
            'student_id': self.__student_id,
            'name': self.__name,
            'email': self.__email,
            'gpa': self.__gpa,
            'enrollment_count': len(self.__enrolled_courses),
            'max_enrollment': self.__max_enrollment,
            'is_active': self.__is_active
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"SecureStudent({self.__student_id}: {self.__name})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"SecureStudentRecord('{self.__student_id}', '{self.__name}', '{self.__email}')"
