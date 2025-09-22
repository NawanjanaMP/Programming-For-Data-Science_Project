"""
Faculty management module demonstrating polymorphism and method overriding
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
from person import Faculty, Professor, Lecturer, TA


class FacultyManager:
    """Faculty management system demonstrating polymorphism"""
    
    def __init__(self):
        self.faculty_list: List[Faculty] = []
    
    def add_faculty(self, faculty: Faculty):
        """Add faculty member to the system"""
        self.faculty_list.append(faculty)
        print(f"Added {faculty.name} as {faculty.__class__.__name__}")
    
    def demonstrate_polymorphism(self):
        """Demonstrate polymorphic behavior across different faculty types"""
        print("\n=== Polymorphism Demonstration ===")
        
        for faculty in self.faculty_list:
            print(f"\n{faculty.name} ({faculty.__class__.__name__}):")
            print("Responsibilities:")
            for resp in faculty.get_responsibilities():
                print(f"  - {resp}")
            print(f"Workload: {faculty.calculate_workload()} hours/week")
    
    def get_faculty_by_type(self, faculty_type: type) -> List[Faculty]:
        """Get all faculty members of a specific type"""
        return [f for f in self.faculty_list if isinstance(f, faculty_type)]
    
    def calculate_total_workload(self) -> Dict[str, int]:
        """Calculate total workload by faculty type"""
        workload_by_type = {}
        for faculty in self.faculty_list:
            faculty_type = faculty.__class__.__name__
            if faculty_type not in workload_by_type:
                workload_by_type[faculty_type] = 0
            workload_by_type[faculty_type] += faculty.calculate_workload()
        return workload_by_type
    
    def get_faculty_responsibilities_report(self) -> Dict[str, List[str]]:
        """Generate a report of responsibilities by faculty type"""
        responsibilities_report = {}
        for faculty in self.faculty_list:
            faculty_type = faculty.__class__.__name__
            if faculty_type not in responsibilities_report:
                responsibilities_report[faculty_type] = faculty.get_responsibilities()
        return responsibilities_report


class EnhancedProfessor(Professor):
    """Enhanced Professor class with additional features"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 department: str = "", hire_date: str = "", research_area: str = "",
                 tenure_status: str = "Assistant"):
        super().__init__(person_id, name, email, phone, department, hire_date, research_area)
        self._tenure_status = tenure_status  # Assistant, Associate, Full
        self._publications = []
        self._research_grants = []
        self._committees = []
    
    @property
    def tenure_status(self) -> str:
        return self._tenure_status
    
    def add_publication(self, title: str, journal: str, year: int):
        """Add a publication to professor's record"""
        self._publications.append({
            'title': title,
            'journal': journal,
            'year': year
        })
    
    def add_research_grant(self, title: str, amount: float, agency: str):
        """Add research grant to professor's record"""
        self._research_grants.append({
            'title': title,
            'amount': amount,
            'agency': agency
        })
    
    def join_committee(self, committee_name: str):
        """Join a university committee"""
        if committee_name not in self._committees:
            self._committees.append(committee_name)
    
    def calculate_workload(self) -> int:
        """Enhanced workload calculation including service"""
        base_workload = super().calculate_workload()
        service_load = len(self._committees) * 2  # 2 hours per committee
        return base_workload + service_load
    
    def get_responsibilities(self) -> List[str]:
        """Enhanced responsibilities including service"""
        responsibilities = super().get_responsibilities()
        if self._committees:
            responsibilities.append("Serve on university committees")
        if len(self._publications) > 5:
            responsibilities.append("Mentor junior faculty")
        return responsibilities
    
    def get_research_profile(self) -> Dict[str, Any]:
        """Get comprehensive research profile"""
        return {
            'research_area': self.research_area,
            'publications': len(self._publications),
            'grants': len(self._research_grants),
            'total_funding': sum(g['amount'] for g in self._research_grants),
            'committees': self._committees.copy(),
            'phd_students': len(self._phd_students)
        }


class EnhancedLecturer(Lecturer):
    """Enhanced Lecturer class with teaching focus"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 department: str = "", hire_date: str = "", specialization: str = ""):
        super().__init__(person_id, name, email, phone, department, hire_date)
        self._specialization = specialization
        self._courses_developed = []
        self._teaching_awards = []
        self._student_evaluations = []
    
    @property
    def specialization(self) -> str:
        return self._specialization
    
    def develop_course(self, course_id: str, course_name: str):
        """Add a course developed by this lecturer"""
        self._courses_developed.append({
            'course_id': course_id,
            'course_name': course_name
        })
    
    def add_teaching_award(self, award_name: str, year: int):
        """Add teaching award to lecturer's record"""
        self._teaching_awards.append({
            'award': award_name,
            'year': year
        })
    
    def add_evaluation(self, semester: str, rating: float):
        """Add student evaluation rating"""
        if 1.0 <= rating <= 5.0:
            self._student_evaluations.append({
                'semester': semester,
                'rating': rating
            })
    
    def calculate_workload(self) -> int:
        """Lecturer workload with course development consideration"""
        base_workload = super().calculate_workload()
        development_load = len(self._courses_developed) * 5  # Extra hours for course dev
        return base_workload + development_load
    
    def get_responsibilities(self) -> List[str]:
        """Enhanced lecturer responsibilities"""
        responsibilities = super().get_responsibilities()
        if self._courses_developed:
            responsibilities.append("Develop new courses and curricula")
        if len(self._teaching_awards) > 0:
            responsibilities.append("Share teaching best practices")
        return responsibilities
    
    def get_teaching_profile(self) -> Dict[str, Any]:
        """Get comprehensive teaching profile"""
        avg_evaluation = 0.0
        if self._student_evaluations:
            avg_evaluation = sum(e['rating'] for e in self._student_evaluations) / len(self._student_evaluations)
        
        return {
            'specialization': self._specialization,
            'courses_developed': len(self._courses_developed),
            'teaching_awards': len(self._teaching_awards),
            'average_evaluation': round(avg_evaluation, 2),
            'total_evaluations': len(self._student_evaluations)
        }


class EnhancedTA(TA):
    """Enhanced Teaching Assistant class"""
    
    def __init__(self, person_id: str, name: str, email: str, phone: str = "",
                 department: str = "", hire_date: str = "", supervisor_id: str = "",
                 degree_program: str = "Masters", academic_year: int = 1):
        super().__init__(person_id, name, email, phone, department, hire_date, supervisor_id)
        self._degree_program = degree_program
        self._academic_year = academic_year
        self._lab_sessions = []
        self._office_hours_per_week = 4
        self._students_mentored = []
    
    @property
    def degree_program(self) -> str:
        return self._degree_program
    
    @property
    def academic_year(self) -> int:
        return self._academic_year
    
    def assign_lab_session(self, course_id: str, session_time: str, capacity: int):
        """Assign a lab session to the TA"""
        self._lab_sessions.append({
            'course_id': course_id,
            'session_time': session_time,
            'capacity': capacity
        })
    
    def mentor_student(self, student_id: str):
        """Add a student to mentoring list"""
        if student_id not in self._students_mentored:
            self._students_mentored.append(student_id)
    
    def set_office_hours(self, hours_per_week: int):
        """Set weekly office hours"""
        if 2 <= hours_per_week <= 10:
            self._office_hours_per_week = hours_per_week
    
    def calculate_workload(self) -> int:
        """TA workload including lab sessions and mentoring"""
        base_workload = super().calculate_workload()
        lab_workload = len(self._lab_sessions) * 3  # 3 hours per lab session
        mentoring_workload = len(self._students_mentored) * 1  # 1 hour per student
        office_hours_workload = self._office_hours_per_week
        
        return base_workload + lab_workload + mentoring_workload + office_hours_workload
    
    def get_responsibilities(self) -> List[str]:
        """Enhanced TA responsibilities"""
        responsibilities = super().get_responsibilities()
        if self._lab_sessions:
            responsibilities.append("Conduct laboratory sessions")
        if self._students_mentored:
            responsibilities.append("Mentor undergraduate students")
        responsibilities.append(f"Hold {self._office_hours_per_week} office hours per week")
        return responsibilities
    
    def get_ta_profile(self) -> Dict[str, Any]:
        """Get comprehensive TA profile"""
        return {
            'degree_program': self._degree_program,
            'academic_year': self._academic_year,
            'supervisor_id': self.supervisor_id,
            'lab_sessions': len(self._lab_sessions),
            'students_mentored': len(self._students_mentored),
            'office_hours_per_week': self._office_hours_per_week,
            'max_courses': self._max_courses
        }


class PolymorphismDemo:
    """Class to demonstrate polymorphic behavior"""
    
    @staticmethod
    def demonstrate_method_overriding():
        """Demonstrate method overriding with different faculty types"""
        print("\n=== Method Overriding Demonstration ===")
        
        # Create different faculty types
        faculty_members = [
            EnhancedProfessor("P001", "Dr. Alice Smith", "alice@uni.edu", 
                            department="Computer Science", research_area="AI"),
            EnhancedLecturer("L001", "Prof. Bob Johnson", "bob@uni.edu",
                           department="Mathematics", specialization="Calculus"),
            EnhancedTA("T001", "Charlie Brown", "charlie@uni.edu",
                     degree_program="PhD", academic_year=2)
        ]
        
        # Add some data
        faculty_members[0].add_publication("AI in Education", "AI Journal", 2024)
        faculty_members[0].join_committee("Curriculum Committee")
        
        faculty_members[1].develop_course("MATH101", "Calculus I")
        faculty_members[1].add_teaching_award("Excellence in Teaching", 2023)
        
        faculty_members[2].assign_lab_session("CS101", "Monday 2-4 PM", 25)
        faculty_members[2].mentor_student("S001")
        
        # Demonstrate polymorphism
        for faculty in faculty_members:
            print(f"\n--- {faculty.name} ({faculty.__class__.__name__}) ---")
            
            # Same method call, different behavior
            print("Responsibilities:")
            for resp in faculty.get_responsibilities():
                print(f"  • {resp}")
            
            print(f"Workload: {faculty.calculate_workload()} hours/week")
            
            # Type-specific information
            if isinstance(faculty, EnhancedProfessor):
                profile = faculty.get_research_profile()
                print(f"Research Profile: {profile}")
            elif isinstance(faculty, EnhancedLecturer):
                profile = faculty.get_teaching_profile()
                print(f"Teaching Profile: {profile}")
            elif isinstance(faculty, EnhancedTA):
                profile = faculty.get_ta_profile()
                print(f"TA Profile: {profile}")
    
    @staticmethod
    def demonstrate_list_processing():
        """Demonstrate processing different faculty types in lists"""
        print("\n=== List Processing Demonstration ===")
        
        faculty_list = [
            EnhancedProfessor("P002", "Dr. Sarah Wilson", "sarah@uni.edu"),
            EnhancedLecturer("L002", "Prof. Mike Davis", "mike@uni.edu"),
            EnhancedTA("T002", "Emma White", "emma@uni.edu"),
            EnhancedProfessor("P003", "Dr. John Lee", "john@uni.edu"),
        ]
        
        print("Processing mixed faculty list:")
        
        total_workload = 0
        faculty_count_by_type = {}
        
        for faculty in faculty_list:
            # Polymorphic method calls
            workload = faculty.calculate_workload()
            total_workload += workload
            
            faculty_type = faculty.__class__.__name__
            faculty_count_by_type[faculty_type] = faculty_count_by_type.get(faculty_type, 0) + 1
            
            print(f"{faculty.name}: {workload} hours ({faculty_type})")
        
        print(f"\nSummary:")
        print(f"Total Faculty: {len(faculty_list)}")
        print(f"Total Workload: {total_workload} hours")
        print("Faculty by Type:")
        for ftype, count in faculty_count_by_type.items():
            print(f"  {ftype}: {count}")


def main():
    """Main function to demonstrate faculty management and polymorphism"""
    print("Faculty Management System with Polymorphism")
    print("=" * 50)
    
    # Create faculty manager
    manager = FacultyManager()
    
    # Add different types of faculty
    prof = EnhancedProfessor("P001", "Dr. Alice Smith", "alice@uni.edu", 
                           department="Computer Science", research_area="Machine Learning")
    prof.add_publication("Deep Learning in Education", "AI Journal", 2024)
    prof.join_committee("Curriculum Committee")
    
    lecturer = EnhancedLecturer("L001", "Prof. Bob Johnson", "bob@uni.edu",
                              department="Mathematics", specialization="Statistics")
    lecturer.develop_course("STAT101", "Introduction to Statistics")
    lecturer.add_teaching_award("Outstanding Teacher Award", 2023)
    
    ta = EnhancedTA("T001", "Charlie Brown", "charlie@uni.edu",
                   degree_program="PhD", academic_year=2)
    ta.assign_lab_session("CS101", "Monday 2-4 PM", 25)
    ta.mentor_student("S001")
    
    # Add to manager
    manager.add_faculty(prof)
    manager.add_faculty(lecturer)
    manager.add_faculty(ta)
    
    # Demonstrate polymorphism
    manager.demonstrate_polymorphism()
    
    # Additional demonstrations
    PolymorphismDemo.demonstrate_method_overriding()
    PolymorphismDemo.demonstrate_list_processing()
    
    # Workload analysis
    print(f"\n=== Workload Analysis ===")
    workload_by_type = manager.calculate_total_workload()
    for faculty_type, total_workload in workload_by_type.items():
        print(f"{faculty_type}: {total_workload} total hours")


if __name__ == "__main__":
    main()