#!/usr/bin/env python3
"""
Main application for the University Management System
Demonstrates all OOP concepts: Inheritance, Encapsulation, Polymorphism, and Abstraction
"""

import sys
from datetime import datetime
from typing import List, Dict

# Import all modules
from person import (
    Person, Student, Faculty, Staff,
    Professor, Lecturer, TA,
    UndergraduateStudent, GraduateStudent
)
from student import EnhancedStudent, SecureStudentRecord
from faculty import EnhancedProfessor, EnhancedLecturer, EnhancedTA, FacultyManager
from department import Department, Course, RegistrationSystem


class UniversityManagementSystem:
    """Main University Management System class"""
    
    def __init__(self, university_name: str = "Tech University"):
        """
        Initialize the University Management System
        
        Args:
            university_name (str): Name of the university
        """
        self.university_name = university_name
        self.registration_system = RegistrationSystem()
        self.faculty_manager = FacultyManager()
        self.students = {}  # {student_id: Student object}
        self.secure_records = {}  # {student_id: SecureStudentRecord}
        
        print(f"Initializing {university_name} Management System...")
        print("=" * 60)
    
    def setup_university(self):
        """Set up the university with departments, courses, faculty, and students"""
        print("\n🏛️  Setting up University Structure...")
        
        # Create Departments
        self._create_departments()
        
        # Create and assign faculty
        self._create_faculty()
        
        # Create courses and set up prerequisites
        self._create_courses()
        
        # Create students
        self._create_students()
        
        print("✅ University setup completed!")
    
    def _create_departments(self):
        """Create university departments"""
        departments_data = [
            ("CS", "Computer Science", "Engineering Building"),
            ("MATH", "Mathematics", "Science Building"),
            ("PHYS", "Physics", "Science Building"),
            ("ENG", "English", "Liberal Arts Building")
        ]
        
        for dept_id, dept_name, building in departments_data:
            dept = Department(dept_id, dept_name, building)
            self.registration_system.add_department(dept)
            print(f"  📚 Created {dept_name} Department")
    
    def _create_faculty(self):
        """Create and assign faculty members"""
        print("\n👥 Creating Faculty Members...")
        
        # Create Professors
        prof1 = EnhancedProfessor(
            "P001", "Dr. Milly Nathalie", "milly.nathalie@techuni.edu",
            "555-0101", "Computer Science", "2020-12-08", 
            "Machine Learning", "Full"
        )
        prof1.add_publication("Deep Learning in Education", "AI Journal", 2024)
        prof1.add_research_grant("AI Education Tools", 250000, "NSF")
        prof1.join_committee("Curriculum Committee")
        
        prof2 = EnhancedProfessor(
            "P002", "Dr. Robert Chen", "robert.chen@techuni.edu",
            "555-0102", "Mathematics", "2018-01-10",
            "Applied Mathematics", "Associate"
        )
        prof2.add_publication("Advanced Calculus Methods", "Math Review", 2023)
        
        # Create Lecturers
        lecturer1 = EnhancedLecturer(
            "L001", "Prof. Maria Garcia", "maria.garcia@techuni.edu",
            "555-0201", "Computer Science", "2021-09-01", "Software Engineering"
        )
        lecturer1.develop_course("CS102", "Object-Oriented Programming")
        lecturer1.add_teaching_award("Excellence in Teaching", 2023)
        
        lecturer2 = EnhancedLecturer(
            "L002", "Prof. David Wilson", "david.wilson@techuni.edu",
            "555-0202", "English", "2019-08-20", "Technical Writing"
        )
        
        # Create Teaching Assistants
        ta1 = EnhancedTA(
            "T001", "Sarah Kim", "sarah.kim@techuni.edu",
            "555-0301", "Computer Science", "2024-01-15", "P001",
            "PhD", 2
        )
        ta1.assign_lab_session("CS101", "Monday 2:00-4:00 PM", 20)
        ta1.mentor_student("S001")
        
        ta2 = EnhancedTA(
            "T002", "Michael Brown", "michael.brown@techuni.edu",
            "555-0302", "Mathematics", "2024-01-15", "P002",
            "Masters", 1
        )
        
        # Add faculty to departments
        cs_dept = self.registration_system.departments["CS"]
        math_dept = self.registration_system.departments["MATH"]
        eng_dept = self.registration_system.departments["ENG"]
        
        cs_dept.add_faculty(prof1)
        cs_dept.add_faculty(lecturer1)
        cs_dept.add_faculty(ta1)
        
        math_dept.add_faculty(prof2)
        math_dept.add_faculty(ta2)
        
        eng_dept.add_faculty(lecturer2)
        
        # Add to faculty manager for polymorphism demos
        self.faculty_manager.add_faculty(prof1)
        self.faculty_manager.add_faculty(prof2)
        self.faculty_manager.add_faculty(lecturer1)
        self.faculty_manager.add_faculty(lecturer2)
        self.faculty_manager.add_faculty(ta1)
        self.faculty_manager.add_faculty(ta2)
        
        # Set department heads
        cs_dept.set_department_head("P001")
        math_dept.set_department_head("P002")
    
    def _create_courses(self):
        """Create courses and set up prerequisites"""
        print("\n📖 Creating Courses...")
        
        courses_data = [
            # Computer Science courses
            ("CS101", "Introduction to Programming", 3, 30, "CS", []),
            ("CS102", "Object-Oriented Programming", 3, 25, "CS", ["CS101"]),
            ("CS201", "Data Structures", 3, 25, "CS", ["CS102"]),
            ("CS301", "Algorithms", 3, 20, "CS", ["CS201", "MATH201"]),
            ("CS401", "Software Engineering", 3, 20, "CS", ["CS301"]),
            
            # Mathematics courses
            ("MATH101", "Calculus I", 4, 40, "MATH", []),
            ("MATH102", "Calculus II", 4, 35, "MATH", ["MATH101"]),
            ("MATH201", "Discrete Mathematics", 3, 30, "MATH", ["MATH101"]),
            ("MATH301", "Linear Algebra", 3, 25, "MATH", ["MATH102"]),
            
            # English courses
            ("ENG101", "English Composition", 3, 25, "ENG", []),
            ("ENG201", "Technical Writing", 3, 20, "ENG", ["ENG101"]),
        ]
        
        for course_id, name, credits, max_enroll, dept_id, prereqs in courses_data:
            course = Course(course_id, name, credits, max_enroll, 
                          self.registration_system.departments[dept_id].dept_name)
            
            # Add prerequisites
            for prereq in prereqs:
                course.add_prerequisite(prereq)
            
            # Add to department
            dept = self.registration_system.departments[dept_id]
            dept.add_course(course)
            
            print(f"  📚 {course_id}: {name} (Prerequisites: {prereqs if prereqs else 'None'})")
        
        # Assign faculty to courses
        assignments = [
            ("P001", "CS301"), ("P001", "CS401"),
            ("L001", "CS101"), ("L001", "CS102"),
            ("T001", "CS101"),  # TA assists with CS101
            ("P002", "MATH101"), ("P002", "MATH201"),
            ("T002", "MATH101"),  # TA assists with MATH101
            ("L002", "ENG101"), ("L002", "ENG201")
        ]
        
        for faculty_id, course_id in assignments:
            for dept in self.registration_system.departments.values():
                if faculty_id in dept._faculty_members:
                    dept.assign_faculty_to_course(faculty_id, course_id)
                    break
    
    def _create_students(self):
        """Create student records"""
        print("\n👨‍🎓 Creating Students...")
        
        # Create different types of students
        students_data = [
            # Undergraduate students
            ("S001", "John Smith", "john.smith@student.techuni.edu", "Undergraduate", 
             "Computer Science", 2, ["MATH101", "CS101"]),
            ("S002", "Emma Johnson", "emma.johnson@student.techuni.edu", "Undergraduate",
             "Computer Science", 3, ["MATH101", "MATH102", "CS101", "CS102", "CS201"]),
            ("S003", "Michael Davis", "michael.davis@student.techuni.edu", "Undergraduate",
             "Mathematics", 2, ["MATH101", "MATH102", "ENG101"]),
            
            # Graduate students
            ("G001", "Lisa Chen", "lisa.chen@student.techuni.edu", "Graduate",
             "Computer Science", 1, ["MATH101", "MATH102", "MATH201"], "P001", "Masters"),
            ("G002", "David Rodriguez", "david.rodriguez@student.techuni.edu", "Graduate",
             "Mathematics", 2, ["MATH101", "MATH102", "MATH201", "MATH301"], "P002", "PhD"),
        ]
        
        for student_data in students_data:
            if student_data[3] == "Undergraduate":
                student_id, name, email, _, major, year, completed = student_data
                
                # Create UndergraduateStudent
                student = UndergraduateStudent(student_id, name, email, "", major, year)
                
            else:  # Graduate
                student_id, name, email, _, major, year, completed, advisor, degree = student_data
                
                # Create GraduateStudent
                student = GraduateStudent(student_id, name, email, "", major, year, 
                                        advisor, degree)
                if degree == "PhD":
                    student.thesis_topic = f"Advanced {major} Research"
            
            # Create enhanced student record
            enhanced_student = EnhancedStudent(student_id, name, email, major)
            
            # Add completed courses and grades
            for course_id in completed:
                enhanced_student.enroll_course(course_id, "Previous Semester", 3)
                # Add random grades between 2.5 and 4.0
                import random
                grade = round(random.uniform(2.5, 4.0), 2)
                enhanced_student.add_grade(course_id, grade)
            
            # Create secure record
            secure_record = SecureStudentRecord(student_id, name, email)
            secure_record.set_gpa(enhanced_student.calculate_gpa())
            
            # Store records
            self.students[student_id] = student
            self.secure_records[student_id] = secure_record
            
            # Update registration system
            self.registration_system.update_student_record(student_id, completed)
            
            print(f"  🎓 {name} ({student_id}) - {student_data[3]} {major}")
    
    def demonstrate_inheritance(self):
        """Demonstrate inheritance hierarchy"""
        print("\n" + "="*60)
        print("🧬 INHERITANCE DEMONSTRATION")
        print("="*60)
        
        print("\n1. Person Hierarchy:")
        for student_id, student in self.students.items():
            print(f"   {student.name}: {student.__class__.__name__} → {student.__class__.__bases__[0].__name__}")
        
        print("\n2. Faculty Hierarchy:")
        for faculty in self.faculty_manager.faculty_list:
            base_classes = " → ".join([cls.__name__ for cls in faculty.__class__.__mro__[::-1]][1:])
            print(f"   {faculty.name}: {base_classes}")
    
    def demonstrate_encapsulation(self):
        """Demonstrate encapsulation with validation"""
        print("\n" + "="*60)
        print("🔒 ENCAPSULATION DEMONSTRATION")
        print("="*60)
        
        student_id = "S001"
        secure_record = self.secure_records[student_id]
        
        print(f"\n1. Testing SecureStudentRecord for {secure_record.name}:")
        print(f"   Current GPA: {secure_record.gpa}")
        
        # Test validation
        print("\n2. Validation Tests:")
        try:
            secure_record.name = ""  # Should fail
        except ValueError as e:
            print(f"   ❌ Invalid name: {e}")
        
        try:
            secure_record.email = "invalid-email"  # Should fail
        except ValueError as e:
            print(f"   ❌ Invalid email: {e}")
        
        try:
            secure_record.set_gpa(5.0)  # Should fail
        except ValueError as e:
            print(f"   ❌ Invalid GPA: {e}")
        
        # Test enrollment limits
        print(f"\n3. Enrollment Limits (Max: 6 courses):")
        test_courses = ["CS102", "MATH201", "ENG101", "CS201", "MATH301", "ENG201", "CS301"]
        for course in test_courses:
            success = secure_record.enroll_in_course(course)
            status = "✅" if success else "❌"
            print(f"   {status} {course}: {secure_record.get_enrollment_count()}/6 courses")
    
    def demonstrate_polymorphism(self):
        """Demonstrate polymorphism"""
        print("\n" + "="*60)
        print("🔄 POLYMORPHISM DEMONSTRATION")
        print("="*60)
        
        print("\n1. Faculty Responsibilities (Same method, different behavior):")
        for faculty in self.faculty_manager.faculty_list[:3]:  # Show first 3
            print(f"\n   {faculty.name} ({faculty.__class__.__name__}):")
            responsibilities = faculty.get_responsibilities()
            for i, resp in enumerate(responsibilities[:3], 1):  # Show first 3 responsibilities
                print(f"      {i}. {resp}")
        
        print("\n2. Workload Calculation (Method overriding):")
        for faculty in self.faculty_manager.faculty_list:
            workload = faculty.calculate_workload()
            print(f"   {faculty.name} ({faculty.__class__.__name__}): {workload} hours/week")
        
        print("\n3. Processing Mixed Lists:")
        all_people = list(self.students.values()) + self.faculty_manager.faculty_list
        person_types = {}
        
        for person in all_people:
            person_type = person.__class__.__name__
            person_types[person_type] = person_types.get(person_type, 0) + 1
        
        print("   Person type distribution:")
        for ptype, count in person_types.items():
            print(f"      {ptype}: {count}")
    
    def demonstrate_advanced_features(self):
        """Demonstrate advanced student management"""
        print("\n" + "="*60)
        print("🚀 ADVANCED STUDENT MANAGEMENT")
        print("="*60)
        
        student_id = "S002"
        enhanced_student = None
        
        # Find the enhanced student
        for sid, student in self.students.items():
            if sid == student_id:
                # Create enhanced student from regular student data
                enhanced_student = EnhancedStudent(student_id, student.name, student.email, student.major)
                break
        
        if enhanced_student:
            print(f"\n1. Student Profile: {enhanced_student.name}")
            
            # Enroll in current semester courses
            current_courses = [("CS301", "Fall 2024"), ("MATH201", "Fall 2024"), ("ENG101", "Fall 2024")]
            
            for course_id, semester in current_courses:
                success = enhanced_student.enroll_course(course_id, semester, 3)
                print(f"   📚 {course_id}: {'✅ Enrolled' if success else '❌ Failed'}")
            
            # Add grades for demonstration
            grade_data = [("CS301", 3.8), ("MATH201", 3.5), ("ENG101", 3.9)]
            for course_id, grade in grade_data:
                enhanced_student.add_grade(course_id, grade)
            
            # Show transcript
            transcript = enhanced_student.get_transcript()
            print(f"\n2. Academic Standing:")
            print(f"   Overall GPA: {transcript['overall_gpa']:.2f}")
            print(f"   Total Credits: {transcript['total_credits']}")
            print(f"   Academic Status: {transcript['academic_status']}")
    
    def demonstrate_course_registration(self):
        """Demonstrate course registration with prerequisites"""
        print("\n" + "="*60)
        print("📝 COURSE REGISTRATION SYSTEM")
        print("="*60)
        
        test_cases = [
            ("S001", "CS102", "Should succeed - has CS101 prerequisite"),
            ("S003", "CS101", "Should succeed - no prerequisites"),
            ("S003", "CS301", "Should fail - missing CS102 and MATH201 prerequisites"),
            ("G001", "CS401", "Should fail - missing CS301 prerequisite"),
        ]
        
        print("\n1. Registration Tests:")
        for student_id, course_id, description in test_cases:
            success, message = self.registration_system.register_student(student_id, course_id)
            status = "✅" if success else "❌"
            student_name = self.students[student_id].name
            print(f"   {status} {student_name} → {course_id}: {message}")
            print(f"      ({description})")
        
        # Show successful enrollments
        print("\n2. Current Course Enrollments:")
        for course_id, course in self.registration_system.all_courses.items():
            if course.current_enrollment > 0:
                print(f"   {course_id}: {course.current_enrollment}/{course.max_enrollment} students")
    
    def generate_reports(self):
        """Generate comprehensive system reports"""
        print("\n" + "="*60)
        print("📊 UNIVERSITY SYSTEM REPORTS")
        print("="*60)
        
        # Faculty workload report
        print("\n1. Faculty Workload Analysis:")
        workload_by_type = self.faculty_manager.calculate_total_workload()
        for faculty_type, total_hours in workload_by_type.items():
            count = len(self.faculty_manager.get_faculty_by_type(eval(faculty_type)))
            avg_hours = total_hours / count if count > 0 else 0
            print(f"   {faculty_type}: {total_hours} total hrs, {avg_hours:.1f} avg hrs/week")
        
        # Department statistics
        print("\n2. Department Statistics:")
        for dept_id, dept in self.registration_system.departments.items():
            stats = dept.get_department_stats()
            print(f"   {stats['department_name']}:")
            print(f"      Faculty: {stats['faculty_count']}, Courses: {stats['course_count']}")
            print(f"      Enrollment: {stats['total_enrollment']}")
            print(f"      Capacity Utilization: {stats['utilization_rate']:.1f}%")
        
        # Registration system report
        print("\n3. Registration System Summary:")
        report = self.registration_system.get_registration_report()
        print(f"   Total Students: {report['total_students']}")
        print(f"   Total Courses: {report['total_courses']}")
        print(f"   Total Enrollments: {report['total_enrollment']}")
        print(f"   Students on Waitlists: {report['total_waitlist']}")
    
    def run_interactive_demo(self):
        """Run interactive demonstration"""
        print("\n" + "="*60)
        print("🎯 INTERACTIVE UNIVERSITY SYSTEM DEMO")
        print("="*60)
        
        while True:
            print("\nSelect a demonstration:")
            print("1. 🧬 Inheritance Hierarchy")
            print("2. 🔒 Encapsulation & Validation")
            print("3. 🔄 Polymorphism")
            print("4. 🚀 Advanced Student Management")
            print("5. 📝 Course Registration")
            print("6. 📊 System Reports")
            print("7. 🎪 Run All Demonstrations")
            print("0. ❌ Exit")
            
            try:
                choice = input("\nEnter your choice (0-7): ").strip()
                
                if choice == "0":
                    print("\n👋 Thank you for using the University Management System!")
                    break
                elif choice == "1":
                    self.demonstrate_inheritance()
                elif choice == "2":
                    self.demonstrate_encapsulation()
                elif choice == "3":
                    self.demonstrate_polymorphism()
                elif choice == "4":
                    self.demonstrate_advanced_features()
                elif choice == "5":
                    self.demonstrate_course_registration()
                elif choice == "6":
                    self.generate_reports()
                elif choice == "7":
                    self.run_all_demonstrations()
                else:
                    print("❌ Invalid choice. Please enter a number between 0-7.")
                
                if choice != "0":
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
    
    def run_all_demonstrations(self):
        """Run all demonstrations in sequence"""
        demonstrations = [
            ("🧬 Inheritance", self.demonstrate_inheritance),
            ("🔒 Encapsulation", self.demonstrate_encapsulation),
            ("🔄 Polymorphism", self.demonstrate_polymorphism),
            ("🚀 Advanced Features", self.demonstrate_advanced_features),
            ("📝 Registration", self.demonstrate_course_registration),
            ("📊 Reports", self.generate_reports)
        ]
        
        for name, demo_func in demonstrations:
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                demo_func()
            except Exception as e:
                print(f"❌ Error in {name}: {e}")


def main():
    """Main entry point for the University Management System"""
    print("🏛️  Welcome to the University Management System!")
    print("This system demonstrates advanced Object-Oriented Programming concepts:")
    print("• Inheritance with multiple class hierarchies")
    print("• Encapsulation with data validation and security")
    print("• Polymorphism with method overriding")
    print("• Abstraction with complex system interactions")
    print("\n" + "="*70)
    
    try:
        # Initialize the system
        ums = UniversityManagementSystem("Tech University")
        
        # Set up the university structure
        ums.setup_university()
        
        # Check if running in interactive mode
        if len(sys.argv) > 1 and sys.argv[1] == "--auto":
            print("\n🚀 Running automated demonstration...")
            ums.run_all_demonstrations()
        else:
            # Run interactive demo
            ums.run_interactive_demo()
    
    except KeyboardInterrupt:
        print("\n\n👋 System shutdown requested. Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("Please check the system configuration and try again.")


if __name__ == "__main__":
    main()