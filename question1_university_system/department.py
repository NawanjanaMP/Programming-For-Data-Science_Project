"""
Department and Course Management System with enrollment and prerequisite checking
"""

from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from person import Faculty, Student


class Course:
    """Course class with enrollment limits and prerequisites"""
    
    def __init__(self, course_id: str, course_name: str, credits: int = 3, 
                 max_enrollment: int = 30, department: str = ""):
        """
        Initialize a course
        
        Args:
            course_id (str): Unique course identifier
            course_name (str): Course name
            credits (int): Credit hours
            max_enrollment (int): Maximum enrollment capacity
            department (str): Department offering the course
        """
        self.course_id = course_id.upper()
        self.course_name = course_name
        self.credits = credits
        self.max_enrollment = max_enrollment
        self.department = department
        
        # Course management
        self._prerequisites = set()  # Set of course IDs
        self._corequisites = set()   # Set of course IDs that must be taken together
        self._enrolled_students = set()  # Set of student IDs
        self._waitlist = []  # List of student IDs on waitlist
        self._assigned_faculty = []  # List of faculty IDs
        
        # Course metadata
        self._description = ""
        self._schedule = {}  # {day: time}
        self._room = ""
        self._semester = ""
        self._is_active = True
    
    @property
    def prerequisites(self) -> Set[str]:
        """Get course prerequisites"""
        return self._prerequisites.copy()
    
    @property
    def corequisites(self) -> Set[str]:
        """Get course corequisites"""
        return self._corequisites.copy()
    
    @property
    def enrolled_students(self) -> Set[str]:
        """Get enrolled student IDs"""
        return self._enrolled_students.copy()
    
    @property
    def assigned_faculty(self) -> List[str]:
        """Get assigned faculty IDs"""
        return self._assigned_faculty.copy()
    
    @property
    def current_enrollment(self) -> int:
        """Get current enrollment count"""
        return len(self._enrolled_students)
    
    @property
    def available_spots(self) -> int:
        """Get available enrollment spots"""
        return max(0, self.max_enrollment - self.current_enrollment)
    
    @property
    def waitlist_count(self) -> int:
        """Get waitlist count"""
        return len(self._waitlist)
    
    def add_prerequisite(self, course_id: str):
        """Add a prerequisite course"""
        self._prerequisites.add(course_id.upper())
    
    def remove_prerequisite(self, course_id: str):
        """Remove a prerequisite course"""
        self._prerequisites.discard(course_id.upper())
    
    def add_corequisite(self, course_id: str):
        """Add a corequisite course"""
        self._corequisites.add(course_id.upper())
    
    def set_schedule(self, schedule: Dict[str, str]):
        """Set course schedule"""
        self._schedule = schedule.copy()
    
    def set_details(self, description: str = "", room: str = "", semester: str = ""):
        """Set additional course details"""
        if description:
            self._description = description
        if room:
            self._room = room
        if semester:
            self._semester = semester
    
    def assign_faculty(self, faculty_id: str) -> bool:
        """
        Assign faculty to course
        
        Args:
            faculty_id (str): Faculty member ID
            
        Returns:
            bool: True if assignment successful
        """
        if faculty_id not in self._assigned_faculty:
            self._assigned_faculty.append(faculty_id)
            return True
        return False
    
    def unassign_faculty(self, faculty_id: str) -> bool:
        """Remove faculty assignment"""
        if faculty_id in self._assigned_faculty:
            self._assigned_faculty.remove(faculty_id)
            return True
        return False
    
    def enroll_student(self, student_id: str) -> Tuple[bool, str]:
        """
        Enroll a student in the course
        
        Args:
            student_id (str): Student ID
            
        Returns:
            Tuple[bool, str]: (Success status, message)
        """
        if not self._is_active:
            return False, "Course is not active"
        
        if student_id in self._enrolled_students:
            return False, "Student already enrolled"
        
        if self.current_enrollment < self.max_enrollment:
            self._enrolled_students.add(student_id)
            # Remove from waitlist if present
            if student_id in self._waitlist:
                self._waitlist.remove(student_id)
            return True, "Successfully enrolled"
        else:
            # Add to waitlist
            if student_id not in self._waitlist:
                self._waitlist.append(student_id)
                return False, f"Course full. Added to waitlist (position {len(self._waitlist)})"
            else:
                return False, "Already on waitlist"
    
    def drop_student(self, student_id: str) -> Tuple[bool, str]:
        """
        Drop a student from the course
        
        Args:
            student_id (str): Student ID
            
        Returns:
            Tuple[bool, str]: (Success status, message)
        """
        if student_id in self._enrolled_students:
            self._enrolled_students.remove(student_id)
            
            # Move waitlist student to enrolled if available
            if self._waitlist and self.available_spots > 0:
                next_student = self._waitlist.pop(0)
                self._enrolled_students.add(next_student)
                return True, f"Dropped successfully. {next_student} moved from waitlist"
            
            return True, "Dropped successfully"
        else:
            return False, "Student not enrolled in course"
    
    def get_course_info(self) -> Dict:
        """Get comprehensive course information"""
        return {
            'course_id': self.course_id,
            'course_name': self.course_name,
            'credits': self.credits,
            'department': self.department,
            'max_enrollment': self.max_enrollment,
            'current_enrollment': self.current_enrollment,
            'available_spots': self.available_spots,
            'waitlist_count': self.waitlist_count,
            'prerequisites': list(self._prerequisites),
            'corequisites': list(self._corequisites),
            'assigned_faculty': self._assigned_faculty.copy(),
            'schedule': self._schedule.copy(),
            'room': self._room,
            'semester': self._semester,
            'description': self._description,
            'is_active': self._is_active
        }
    
    def __str__(self) -> str:
        return f"{self.course_id}: {self.course_name} ({self.current_enrollment}/{self.max_enrollment})"


class Department:
    """Department class managing faculty, courses, and students"""
    
    def __init__(self, dept_id: str, dept_name: str, building: str = ""):
        """
        Initialize department
        
        Args:
            dept_id (str): Department ID
            dept_name (str): Department name
            building (str): Building location
        """
        self.dept_id = dept_id.upper()
        self.dept_name = dept_name
        self.building = building
        
        # Department resources
        self._faculty_members = {}  # {faculty_id: Faculty object}
        self._courses = {}  # {course_id: Course object}
        self._students = {}  # {student_id: Student object}
        
        # Department administration
        self._department_head = None
        self._budget = 0.0
        self._office_location = ""
        self._phone = ""
        self._email = ""
    
    @property
    def faculty_count(self) -> int:
        """Get number of faculty members"""
        return len(self._faculty_members)
    
    @property
    def course_count(self) -> int:
        """Get number of courses offered"""
        return len(self._courses)
    
    @property
    def student_count(self) -> int:
        """Get number of students in department"""
        return len(self._students)
    
    def set_department_head(self, faculty_id: str) -> bool:
        """Set department head"""
        if faculty_id in self._faculty_members:
            self._department_head = faculty_id
            return True
        return False
    
    def add_faculty(self, faculty: Faculty) -> bool:
        """
        Add faculty member to department
        
        Args:
            faculty (Faculty): Faculty object
            
        Returns:
            bool: True if added successfully
        """
        if faculty.person_id not in self._faculty_members:
            self._faculty_members[faculty.person_id] = faculty
            # Update faculty's department
            faculty.department = self.dept_name
            return True
        return False
    
    def remove_faculty(self, faculty_id: str) -> bool:
        """Remove faculty member from department"""
        if faculty_id in self._faculty_members:
            # Remove faculty from all assigned courses
            for course in self._courses.values():
                course.unassign_faculty(faculty_id)
            del self._faculty_members[faculty_id]
            return True
        return False
    
    def add_course(self, course: Course) -> bool:
        """
        Add course to department
        
        Args:
            course (Course): Course object
            
        Returns:
            bool: True if added successfully
        """
        if course.course_id not in self._courses:
            self._courses[course.course_id] = course
            course.department = self.dept_name
            return True
        return False
    
    def remove_course(self, course_id: str) -> bool:
        """Remove course from department"""
        course_id = course_id.upper()
        if course_id in self._courses:
            del self._courses[course_id]
            return True
        return False
    
    def add_student(self, student: Student) -> bool:
        """Add student to department"""
        if student.person_id not in self._students:
            self._students[student.person_id] = student
            return True
        return False
    
    def assign_faculty_to_course(self, faculty_id: str, course_id: str) -> bool:
        """
        Assign faculty member to teach a course
        
        Args:
            faculty_id (str): Faculty ID
            course_id (str): Course ID
            
        Returns:
            bool: True if assignment successful
        """
        course_id = course_id.upper()
        
        if faculty_id not in self._faculty_members:
            print(f"Faculty {faculty_id} not found in department")
            return False
        
        if course_id not in self._courses:
            print(f"Course {course_id} not found in department")
            return False
        
        course = self._courses[course_id]
        success = course.assign_faculty(faculty_id)
        
        if success:
            # Update faculty's assigned courses
            faculty = self._faculty_members[faculty_id]
            if hasattr(faculty, '_assigned_courses'):
                if course_id not in faculty._assigned_courses:
                    faculty._assigned_courses.append(course_id)
        
        return success
    
    def get_faculty_workload(self) -> Dict[str, int]:
        """Get workload for all faculty members"""
        workload = {}
        for faculty_id, faculty in self._faculty_members.items():
            workload[faculty_id] = faculty.calculate_workload()
        return workload
    
    def get_course_enrollments(self) -> Dict[str, int]:
        """Get enrollment counts for all courses"""
        enrollments = {}
        for course_id, course in self._courses.items():
            enrollments[course_id] = course.current_enrollment
        return enrollments
    
    def get_department_stats(self) -> Dict:
        """Get comprehensive department statistics"""
        total_enrollment = sum(course.current_enrollment for course in self._courses.values())
        total_capacity = sum(course.max_enrollment for course in self._courses.values())
        
        faculty_by_type = {}
        for faculty in self._faculty_members.values():
            ftype = faculty.__class__.__name__
            faculty_by_type[ftype] = faculty_by_type.get(ftype, 0) + 1
        
        return {
            'department_id': self.dept_id,
            'department_name': self.dept_name,
            'faculty_count': self.faculty_count,
            'course_count': self.course_count,
            'student_count': self.student_count,
            'total_enrollment': total_enrollment,
            'total_capacity': total_capacity,
            'utilization_rate': (total_enrollment / total_capacity * 100) if total_capacity > 0 else 0,
            'faculty_by_type': faculty_by_type,
            'department_head': self._department_head,
            'building': self.building
        }


class RegistrationSystem:
    """Course registration system with prerequisite checking"""
    
    def __init__(self):
        self.departments = {}  # {dept_id: Department}
        self.all_courses = {}  # {course_id: Course} - for easy lookup
        self.student_records = {}  # {student_id: completed_courses}
    
    def add_department(self, department: Department):
        """Add department to the system"""
        self.departments[department.dept_id] = department
        
        # Add department courses to global course lookup
        for course_id, course in department._courses.items():
            self.all_courses[course_id] = course
    
    def update_student_record(self, student_id: str, completed_courses: List[str]):
        """Update student's completed course record"""
        self.student_records[student_id] = set(course.upper() for course in completed_courses)
    
    def check_prerequisites(self, student_id: str, course_id: str) -> Tuple[bool, List[str]]:
        """
        Check if student meets prerequisites for a course
        
        Args:
            student_id (str): Student ID
            course_id (str): Course ID
            
        Returns:
            Tuple[bool, List[str]]: (Prerequisites met, Missing prerequisites)
        """
        course_id = course_id.upper()
        
        if course_id not in self.all_courses:
            return False, ["Course not found"]
        
        course = self.all_courses[course_id]
        completed_courses = self.student_records.get(student_id, set())
        
        missing_prereqs = []
        for prereq in course.prerequisites:
            if prereq not in completed_courses:
                missing_prereqs.append(prereq)
        
        return len(missing_prereqs) == 0, missing_prereqs
    
    def check_corequisites(self, student_id: str, course_id: str, 
                          current_enrollment: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if student is enrolling in required corequisites
        
        Args:
            student_id (str): Student ID
            course_id (str): Course ID
            current_enrollment (List[str]): Courses student is enrolling in this semester
            
        Returns:
            Tuple[bool, List[str]]: (Corequisites satisfied, Missing corequisites)
        """
        course_id = course_id.upper()
        current_enrollment = [c.upper() for c in current_enrollment]
        
        if course_id not in self.all_courses:
            return False, ["Course not found"]
        
        course = self.all_courses[course_id]
        completed_courses = self.student_records.get(student_id, set())
        
        missing_coreqs = []
        for coreq in course.corequisites:
            if coreq not in completed_courses and coreq not in current_enrollment:
                missing_coreqs.append(coreq)
        
        return len(missing_coreqs) == 0, missing_coreqs
    
    def register_student(self, student_id: str, course_id: str, 
                        current_semester_courses: List[str] = None) -> Tuple[bool, str]:
        """
        Register student for a course with all validation checks
        
        Args:
            student_id (str): Student ID
            course_id (str): Course ID
            current_semester_courses (List[str]): Other courses student is taking
            
        Returns:
            Tuple[bool, str]: (Success status, message)
        """
        course_id = course_id.upper()
        current_semester_courses = current_semester_courses or []
        
        if course_id not in self.all_courses:
            return False, "Course not found"
        
        course = self.all_courses[course_id]
        
        # Check prerequisites
        prereq_met, missing_prereqs = self.check_prerequisites(student_id, course_id)
        if not prereq_met:
            return False, f"Missing prerequisites: {', '.join(missing_prereqs)}"
        
        # Check corequisites
        coreq_met, missing_coreqs = self.check_corequisites(
            student_id, course_id, current_semester_courses
        )
        if not coreq_met:
            return False, f"Missing corequisites: {', '.join(missing_coreqs)}"
        
        # Attempt enrollment
        success, message = course.enroll_student(student_id)
        return success, message
    
    def get_registration_report(self) -> Dict:
        """Generate comprehensive registration report"""
        total_students = len(self.student_records)
        total_courses = len(self.all_courses)
        total_enrollment = sum(course.current_enrollment for course in self.all_courses.values())
        total_waitlist = sum(course.waitlist_count for course in self.all_courses.values())
        
        dept_stats = {}
        for dept_id, dept in self.departments.items():
            dept_stats[dept_id] = dept.get_department_stats()
        
        return {
            'total_students': total_students,
            'total_courses': total_courses,
            'total_enrollment': total_enrollment,
            'total_waitlist': total_waitlist,
            'departments': dept_stats,
            'courses_by_enrollment': {
                course_id: course.current_enrollment 
                for course_id, course in self.all_courses.items()
            }
        }


def main():
    """Demonstrate the department and course management system"""
    print("Department and Course Management System")
    print("=" * 50)
    
    # Create registration system
    reg_system = RegistrationSystem()
    
    # Create departments
    cs_dept = Department("CS", "Computer Science", "Engineering Building")
    math_dept = Department("MATH", "Mathematics", "Science Building")
    
    # Create courses
    # Computer Science courses
    cs101 = Course("CS101", "Introduction to Programming", 3, 25, "Computer Science")
    cs201 = Course("CS201", "Data Structures", 3, 30, "Computer Science")
    cs201.add_prerequisite("CS101")
    
    cs301 = Course("CS301", "Algorithms", 3, 25, "Computer Science")
    cs301.add_prerequisite("CS201")
    cs301.add_prerequisite("MATH201")
    
    # Mathematics courses
    math101 = Course("MATH101", "Calculus I", 4, 40, "Mathematics")
    math201 = Course("MATH201", "Discrete Mathematics", 3, 35, "Mathematics")
    math201.add_prerequisite("MATH101")
    
    # Add courses to departments
    cs_dept.add_course(cs101)
    cs_dept.add_course(cs201)
    cs_dept.add_course(cs301)
    
    math_dept.add_course(math101)
    math_dept.add_course(math201)
    
    # Add departments to registration system
    reg_system.add_department(cs_dept)
    reg_system.add_department(math_dept)
    
    # Create and add faculty (simplified - using basic Faculty class)
    from person import Professor, Lecturer
    
    prof_smith = Professor("P001", "Dr. Alice Smith", "alice@uni.edu", 
                          department="Computer Science", research_area="Algorithms")
    prof_johnson = Professor("P002", "Dr. Bob Johnson", "bob@uni.edu",
                           department="Mathematics", research_area="Discrete Math")
    
    cs_dept.add_faculty(prof_smith)
    math_dept.add_faculty(prof_johnson)
    
    # Assign faculty to courses
    cs_dept.assign_faculty_to_course("P001", "CS101")
    cs_dept.assign_faculty_to_course("P001", "CS301")
    math_dept.assign_faculty_to_course("P002", "MATH101")
    math_dept.assign_faculty_to_course("P002", "MATH201")
    
    # Create student records (completed courses)
    reg_system.update_student_record("S001", ["MATH101"])
    reg_system.update_student_record("S002", ["CS101", "MATH101"])
    reg_system.update_student_record("S003", ["CS101", "MATH101", "MATH201"])
    reg_system.update_student_record("S004", [])  # New student
    
    print("\n=== Course Information ===")
    for course_id, course in reg_system.all_courses.items():
        info = course.get_course_info()
        print(f"\n{course_id}: {info['course_name']}")
        print(f"  Credits: {info['credits']}")
        print(f"  Capacity: {info['current_enrollment']}/{info['max_enrollment']}")
        if info['prerequisites']:
            print(f"  Prerequisites: {', '.join(info['prerequisites'])}")
        if info['assigned_faculty']:
            print(f"  Faculty: {', '.join(info['assigned_faculty'])}")
    
    print("\n=== Registration Demonstrations ===")
    
    # Test successful registration
    print("\n1. Student S001 registering for MATH201 (has MATH101 prerequisite):")
    success, msg = reg_system.register_student("S001", "MATH201")
    print(f"   Result: {msg}")
    
    # Test prerequisite failure
    print("\n2. Student S004 trying to register for CS201 (missing CS101 prerequisite):")
    success, msg = reg_system.register_student("S004", "CS201")
    print(f"   Result: {msg}")
    
    # Test successful multi-prerequisite registration
    print("\n3. Student S003 registering for CS301 (has CS101 and MATH201 prerequisites):")
    success, msg = reg_system.register_student("S003", "CS301")
    print(f"   Result: {msg}")
    
    # Test course capacity
    print("\n4. Testing course capacity limits:")
    # Fill up CS101
    for i in range(25):
        student_id = f"S{100+i:03d}"
        reg_system.update_student_record(student_id, [])
        success, msg = reg_system.register_student(student_id, "CS101")
        if not success and "waitlist" in msg.lower():
            print(f"   Student {student_id}: {msg}")
            break
        elif i == 24:  # Last student to fit
            print(f"   Student {student_id}: {msg}")
    
    # One more student to test waitlist
    reg_system.update_student_record("S200", [])
    success, msg = reg_system.register_student("S200", "CS101")
    print(f"   Student S200: {msg}")
    
    print("\n=== Department Statistics ===")
    for dept_id, dept in reg_system.departments.items():
        stats = dept.get_department_stats()
        print(f"\n{stats['department_name']} Department:")
        print(f"  Faculty: {stats['faculty_count']}")
        print(f"  Courses: {stats['course_count']}")
        print(f"  Total Enrollment: {stats['total_enrollment']}")
        print(f"  Capacity Utilization: {stats['utilization_rate']:.1f}%")
        if stats['faculty_by_type']:
            print("  Faculty by Type:")
            for ftype, count in stats['faculty_by_type'].items():
                print(f"    {ftype}: {count}")
    
    print("\n=== Registration System Report ===")
    report = reg_system.get_registration_report()
    print(f"Total Students: {report['total_students']}")
    print(f"Total Courses: {report['total_courses']}")
    print(f"Total Enrollments: {report['total_enrollment']}")
    print(f"Total Waitlist: {report['total_waitlist']}")
    
    print("\nCourse Enrollments:")
    for course_id, enrollment in report['courses_by_enrollment'].items():
        course = reg_system.all_courses[course_id]
        print(f"  {course_id}: {enrollment}/{course.max_enrollment}")
    
    print("\n=== Advanced Features Demonstration ===")
    
    # Demonstrate corequisites
    print("\n1. Corequisites Example:")
    lab_course = Course("CS101L", "Programming Lab", 1, 25, "Computer Science")
    lab_course.add_corequisite("CS101")
    cs101.add_corequisite("CS101L")
    
    cs_dept.add_course(lab_course)
    reg_system.all_courses["CS101L"] = lab_course
    
    # Student trying to take CS101 without lab
    success, msg = reg_system.register_student("S201", "CS101", [])
    print(f"   S201 taking CS101 alone: {msg}")
    
    # Student taking both CS101 and lab together
    success, msg = reg_system.register_student("S201", "CS101", ["CS101L"])
    print(f"   S201 taking CS101 with lab: {msg}")
    
    # Demonstrate course schedule conflicts (basic implementation)
    print("\n2. Course Scheduling:")
    cs101.set_schedule({"Monday": "10:00-11:30", "Wednesday": "10:00-11:30"})
    cs201.set_schedule({"Tuesday": "14:00-15:30", "Thursday": "14:00-15:30"})
    
    print("   CS101 Schedule:", cs101._schedule)
    print("   CS201 Schedule:", cs201._schedule)


if __name__ == "__main__":
    main()