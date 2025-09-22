"""
Unit tests for the University Management System
This is a bonus feature demonstrating testing best practices
"""

import unittest
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from person import (
    Person, Student, Faculty, Professor, Lecturer, TA,
    UndergraduateStudent, GraduateStudent
)
from student import EnhancedStudent, SecureStudentRecord
from faculty import EnhancedProfessor, EnhancedLecturer, EnhancedTA
from department import Department, Course, RegistrationSystem


class TestPersonHierarchy(unittest.TestCase):
    """Test the Person class hierarchy and inheritance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.undergrad = UndergraduateStudent("S001", "John Doe", "john@test.edu", 
                                             major="Computer Science", year=2)
        self.grad = GraduateStudent("G001", "Jane Smith", "jane@test.edu",
                                   major="Mathematics", advisor_id="P001", degree_type="Masters")
        self.professor = Professor("P001", "Dr. Alice Johnson", "alice@test.edu",
                                  department="Computer Science", research_area="AI")
        self.lecturer = Lecturer("L001", "Prof. Bob Wilson", "bob@test.edu")
        self.ta = TA("T001", "Charlie Brown", "charlie@test.edu", supervisor_id="P001")
    
    def test_inheritance_chain(self):
        """Test that inheritance is working correctly"""
        # Test Student inheritance
        self.assertIsInstance(self.undergrad, Student)
        self.assertIsInstance(self.undergrad, Person)
        self.assertIsInstance(self.grad, Student)
        self.assertIsInstance(self.grad, Person)
        
        # Test Faculty inheritance
        self.assertIsInstance(self.professor, Faculty)
        self.assertIsInstance(self.professor, Person)
        self.assertIsInstance(self.lecturer, Faculty)
        self.assertIsInstance(self.lecturer, Person)
        self.assertIsInstance(self.ta, Faculty)
        self.assertIsInstance(self.ta, Person)
    
    def test_method_inheritance(self):
        """Test that methods are inherited properly"""
        # All Person objects should have basic info
        for person in [self.undergrad, self.grad, self.professor, self.lecturer, self.ta]:
            self.assertIsNotNone(person.get_basic_info())
            self.assertIn('name', person.get_basic_info())
            self.assertIn('email', person.get_basic_info())
    
    def test_polymorphic_methods(self):
        """Test polymorphic behavior of get_responsibilities method"""
        # All persons should have responsibilities, but different ones
        undergrad_resp = self.undergrad.get_responsibilities()
        prof_resp = self.professor.get_responsibilities()
        ta_resp = self.ta.get_responsibilities()
        
        self.assertIsInstance(undergrad_resp, list)
        self.assertIsInstance(prof_resp, list)
        self.assertIsInstance(ta_resp, list)
        
        # Responsibilities should be different
        self.assertNotEqual(undergrad_resp, prof_resp)
        self.assertNotEqual(prof_resp, ta_resp)


class TestStudentManagement(unittest.TestCase):
    """Test enhanced student management features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.student = EnhancedStudent("S001", "John Doe", "john@test.edu", "Computer Science")
    
    def test_course_enrollment(self):
        """Test course enrollment functionality"""
        # Test successful enrollment
        result = self.student.enroll_course("CS101", "Fall 2024", 3)
        self.assertTrue(result)
        
        # Test duplicate enrollment
        result = self.student.enroll_course("CS101", "Fall 2024", 3)
        self.assertFalse(result)
        
        # Test credit limit
        for i in range(6):  # Try to enroll in 6 more courses (18 credits total limit)
            self.student.enroll_course(f"CS{200+i}", "Fall 2024", 3)
        
        # This should fail due to credit limit
        result = self.student.enroll_course("CS299", "Fall 2024", 3)
        self.assertFalse(result)
    
    def test_course_drop(self):
        """Test course drop functionality"""
        # Enroll first
        self.student.enroll_course("CS101", "Fall 2024", 3)
        
        # Test successful drop
        result = self.student.drop_course("CS101")
        self.assertTrue(result)
        
        # Test dropping non-enrolled course
        result = self.student.drop_course("CS102")
        self.assertFalse(result)
    
    def test_gpa_calculation(self):
        """Test GPA calculation"""
        # Add some courses and grades
        self.student.enroll_course("CS101", "Fall 2024", 3)
        self.student.enroll_course("MATH101", "Fall 2024", 4)
        
        self.student.add_grade("CS101", 3.5)
        self.student.add_grade("MATH101", 4.0)
        
        # Calculate expected GPA: (3.5*3 + 4.0*4) / (3+4) = 3.79
        expected_gpa = (3.5 * 3 + 4.0 * 4) / (3 + 4)
        actual_gpa = self.student.calculate_gpa()
        
        self.assertAlmostEqual(actual_gpa, expected_gpa, places=2)
    
    def test_academic_status(self):
        """Test academic status calculation"""
        # Add grades for Dean's List (GPA >= 3.5)
        self.student.enroll_course("CS101", "Fall 2024", 3)
        self.student.add_grade("CS101", 3.8)
        
        status = self.student.get_academic_status()
        self.assertEqual(status, "Dean's List")
        
        # Test Good Standing (2.0 <= GPA < 3.5)
        self.student.enroll_course("CS102", "Fall 2024", 3)
        self.student.add_grade("CS102", 2.5)
        
        status = self.student.get_academic_status()
        self.assertEqual(status, "Good Standing")


class TestSecureStudentRecord(unittest.TestCase):
    """Test encapsulation and validation in SecureStudentRecord"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.secure_student = SecureStudentRecord("S001", "John Doe", "john@test.edu")
    
    def test_property_access(self):
        """Test property getter methods"""
        self.assertEqual(self.secure_student.student_id, "S001")
        self.assertEqual(self.secure_student.name, "John Doe")
        self.assertEqual(self.secure_student.email, "john@test.edu")
        self.assertEqual(self.secure_student.gpa, 0.0)
        self.assertTrue(self.secure_student.is_active)
    
    def test_name_validation(self):
        """Test name validation"""
        # Valid names
        self.secure_student.name = "Jane Smith"
        self.assertEqual(self.secure_student.name, "Jane Smith")
        
        # Invalid names
        with self.assertRaises(ValueError):
            self.secure_student.name = ""
        
        with self.assertRaises(ValueError):
            self.secure_student.name = "   "
        
        with self.assertRaises(ValueError):
            self.secure_student.name = "J"  # Too short
        
        with self.assertRaises(ValueError):
            self.secure_student.name = "John123"  # Contains numbers
    
    def test_email_validation(self):
        """Test email validation"""
        # Valid emails
        self.secure_student.email = "test@example.com"
        self.assertEqual(self.secure_student.email, "test@example.com")
        
        # Invalid emails
        with self.assertRaises(ValueError):
            self.secure_student.email = "invalid-email"
        
        with self.assertRaises(ValueError):
            self.secure_student.email = "@example.com"
        
        with self.assertRaises(ValueError):
            self.secure_student.email = "test@"
    
    def test_gpa_validation(self):
        """Test GPA validation"""
        # Valid GPAs
        self.secure_student.set_gpa(3.5)
        self.assertEqual(self.secure_student.gpa, 3.5)
        
        self.secure_student.set_gpa(0.0)
        self.assertEqual(self.secure_student.gpa, 0.0)
        
        self.secure_student.set_gpa(4.0)
        self.assertEqual(self.secure_student.gpa, 4.0)
        
        # Invalid GPAs
        with self.assertRaises(ValueError):
            self.secure_student.set_gpa(-1.0)
        
        with self.assertRaises(ValueError):
            self.secure_student.set_gpa(5.0)
        
        with self.assertRaises(ValueError):
            self.secure_student.set_gpa("3.5")  # String instead of number
    
    def test_enrollment_limits(self):
        """Test course enrollment limits"""
        # Enroll in maximum courses (6)
        for i in range(6):
            result = self.secure_student.enroll_in_course(f"CS{101+i}")
            self.assertTrue(result)
        
        # Try to enroll in one more (should fail)
        result = self.secure_student.enroll_in_course("CS107")
        self.assertFalse(result)
        
        # Check enrollment count
        self.assertEqual(self.secure_student.get_enrollment_count(), 6)


class TestFacultyPolymorphism(unittest.TestCase):
    """Test polymorphism in faculty classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.professor = EnhancedProfessor("P001", "Dr. Smith", "smith@test.edu", 
                                          research_area="AI")
        self.lecturer = EnhancedLecturer("L001", "Prof. Jones", "jones@test.edu",
                                        specialization="Software Engineering")
        self.ta = EnhancedTA("T001", "Alice Brown", "alice@test.edu", 
                            degree_program="PhD")
    
    def test_workload_calculation_polymorphism(self):
        """Test that different faculty types calculate workload differently"""
        # Add some assignments to make workload calculations meaningful
        self.professor._assigned_courses = ["CS301", "CS401"]
        self.lecturer._assigned_courses = ["CS101", "CS102", "CS201"]
        self.ta._assigned_courses = ["CS101"]
        
        prof_workload = self.professor.calculate_workload()
        lecturer_workload = self.lecturer.calculate_workload()
        ta_workload = self.ta.calculate_workload()
        
        # All should return positive integers
        self.assertIsInstance(prof_workload, int)
        self.assertIsInstance(lecturer_workload, int)
        self.assertIsInstance(ta_workload, int)
        
        self.assertGreater(prof_workload, 0)
        self.assertGreater(lecturer_workload, 0)
        self.assertGreater(ta_workload, 0)
        
        # Different types should calculate differently (at least some should be different)
        workloads = [prof_workload, lecturer_workload, ta_workload]
        self.assertGreater(len(set(workloads)), 1, "All workloads are the same, polymorphism not working")
    
    def test_responsibilities_polymorphism(self):
        """Test that different faculty types have different responsibilities"""
        prof_resp = self.professor.get_responsibilities()
        lecturer_resp = self.lecturer.get_responsibilities()
        ta_resp = self.ta.get_responsibilities()
        
        # All should return lists
        self.assertIsInstance(prof_resp, list)
        self.assertIsInstance(lecturer_resp, list)
        self.assertIsInstance(ta_resp, list)
        
        # All should have at least one responsibility
        self.assertGreater(len(prof_resp), 0)
        self.assertGreater(len(lecturer_resp), 0)
        self.assertGreater(len(ta_resp), 0)
        
        # Responsibilities should be different
        self.assertNotEqual(prof_resp, lecturer_resp)
        self.assertNotEqual(lecturer_resp, ta_resp)
        self.assertNotEqual(prof_resp, ta_resp)


class TestCourseManagement(unittest.TestCase):
    """Test course management functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.course = Course("CS101", "Introduction to Programming", 3, 25, "Computer Science")
        self.advanced_course = Course("CS301", "Algorithms", 3, 20, "Computer Science")
        self.advanced_course.add_prerequisite("CS101")
        self.advanced_course.add_prerequisite("MATH201")
    
    def test_course_creation(self):
        """Test course creation and properties"""
        self.assertEqual(self.course.course_id, "CS101")
        self.assertEqual(self.course.course_name, "Introduction to Programming")
        self.assertEqual(self.course.credits, 3)
        self.assertEqual(self.course.max_enrollment, 25)
        self.assertEqual(self.course.current_enrollment, 0)
        self.assertEqual(self.course.available_spots, 25)
    
    def test_prerequisite_management(self):
        """Test prerequisite addition and removal"""
        # Test prerequisites
        prereqs = self.advanced_course.prerequisites
        self.assertIn("CS101", prereqs)
        self.assertIn("MATH201", prereqs)
        
        # Test removing prerequisite
        self.advanced_course.remove_prerequisite("MATH201")
        prereqs = self.advanced_course.prerequisites
        self.assertNotIn("MATH201", prereqs)
        self.assertIn("CS101", prereqs)
    
    def test_student_enrollment(self):
        """Test student enrollment in courses"""
        # Test successful enrollment
        success, message = self.course.enroll_student("S001")
        self.assertTrue(success)
        self.assertEqual(self.course.current_enrollment, 1)
        
        # Test duplicate enrollment
        success, message = self.course.enroll_student("S001")
        self.assertFalse(success)
        
        # Test capacity limits
        # Fill up the course
        for i in range(2, 26):  # Students S002 to S025
            self.course.enroll_student(f"S{i:03d}")
        
        # Course should be full
        self.assertEqual(self.course.current_enrollment, 25)
        self.assertEqual(self.course.available_spots, 0)
        
        # Next student should go to waitlist
        success, message = self.course.enroll_student("S026")
        self.assertFalse(success)
        self.assertIn("waitlist", message.lower())
        self.assertEqual(self.course.waitlist_count, 1)
    
    def test_student_drop(self):
        """Test student dropping from courses"""
        # Enroll student
        self.course.enroll_student("S001")
        
        # Drop student
        success, message = self.course.drop_student("S001")
        self.assertTrue(success)
        self.assertEqual(self.course.current_enrollment, 0)
        
        # Test dropping non-enrolled student
        success, message = self.course.drop_student("S999")
        self.assertFalse(success)


class TestRegistrationSystem(unittest.TestCase):
    """Test the course registration system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reg_system = RegistrationSystem()
        
        # Create courses
        self.cs101 = Course("CS101", "Introduction to Programming", 3, 25, "Computer Science")
        self.cs201 = Course("CS201", "Data Structures", 3, 20, "Computer Science")
        self.cs201.add_prerequisite("CS101")
        
        self.math101 = Course("MATH101", "Calculus I", 4, 30, "Mathematics")
        self.math201 = Course("MATH201", "Discrete Math", 3, 25, "Mathematics")
        self.math201.add_prerequisite("MATH101")
        
        # Create department and add courses
        cs_dept = Department("CS", "Computer Science")
        cs_dept.add_course(self.cs101)
        cs_dept.add_course(self.cs201)
        
        math_dept = Department("MATH", "Mathematics")
        math_dept.add_course(self.math101)
        math_dept.add_course(self.math201)
        
        # Add departments to registration system
        self.reg_system.add_department(cs_dept)
        self.reg_system.add_department(math_dept)
        
        # Set up student records
        self.reg_system.update_student_record("S001", ["CS101", "MATH101"])
        self.reg_system.update_student_record("S002", [])  # New student
    
    def test_prerequisite_checking(self):
        """Test prerequisite validation"""
        # Student with prerequisites should succeed
        prereq_met, missing = self.reg_system.check_prerequisites("S001", "CS201")
        self.assertTrue(prereq_met)
        self.assertEqual(len(missing), 0)
        
        # Student without prerequisites should fail
        prereq_met, missing = self.reg_system.check_prerequisites("S002", "CS201")
        self.assertFalse(prereq_met)
        self.assertIn("CS101", missing)
    
    def test_student_registration(self):
        """Test complete student registration process"""
        # Successful registration with prerequisites
        success, message = self.reg_system.register_student("S001", "CS201")
        self.assertTrue(success)
        
        # Failed registration without prerequisites
        success, message = self.reg_system.register_student("S002", "CS201")
        self.assertFalse(success)
        self.assertIn("prerequisite", message.lower())
        
        # Successful registration without prerequisites
        success, message = self.reg_system.register_student("S002", "CS101")
        self.assertTrue(success)
    
    def test_registration_report(self):
        """Test registration system reporting"""
        # Register some students
        self.reg_system.register_student("S001", "CS201")
        self.reg_system.register_student("S002", "CS101")
        
        report = self.reg_system.get_registration_report()
        
        self.assertIn('total_students', report)
        self.assertIn('total_courses', report)
        self.assertIn('total_enrollment', report)
        self.assertGreater(report['total_enrollment'], 0)


class TestDepartmentManagement(unittest.TestCase):
    """Test department management functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dept = Department("CS", "Computer Science", "Engineering Building")
        self.professor = Professor("P001", "Dr. Smith", "smith@test.edu")
        self.course = Course("CS101", "Programming", 3, 25, "Computer Science")
    
    def test_faculty_management(self):
        """Test adding and removing faculty"""
        # Add faculty
        result = self.dept.add_faculty(self.professor)
        self.assertTrue(result)
        self.assertEqual(self.dept.faculty_count, 1)
        
        # Try to add same faculty again
        result = self.dept.add_faculty(self.professor)
        self.assertFalse(result)
        self.assertEqual(self.dept.faculty_count, 1)
        
        # Remove faculty
        result = self.dept.remove_faculty("P001")
        self.assertTrue(result)
        self.assertEqual(self.dept.faculty_count, 0)
        
        # Try to remove non-existent faculty
        result = self.dept.remove_faculty("P999")
        self.assertFalse(result)
    
    def test_course_management(self):
        """Test adding and removing courses"""
        # Add course
        result = self.dept.add_course(self.course)
        self.assertTrue(result)
        self.assertEqual(self.dept.course_count, 1)
        
        # Try to add same course again
        result = self.dept.add_course(self.course)
        self.assertFalse(result)
        self.assertEqual(self.dept.course_count, 1)
        
        # Remove course
        result = self.dept.remove_course("CS101")
        self.assertTrue(result)
        self.assertEqual(self.dept.course_count, 0)
    
    def test_faculty_course_assignment(self):
        """Test assigning faculty to courses"""
        # Add faculty and course
        self.dept.add_faculty(self.professor)
        self.dept.add_course(self.course)
        
        # Assign faculty to course
        result = self.dept.assign_faculty_to_course("P001", "CS101")
        self.assertTrue(result)
        
        # Verify assignment
        assigned_faculty = self.course.assigned_faculty
        self.assertIn("P001", assigned_faculty)
    
    def test_department_statistics(self):
        """Test department statistics generation"""
        # Add some data
        self.dept.add_faculty(self.professor)
        self.dept.add_course(self.course)
        
        stats = self.dept.get_department_stats()
        
        self.assertEqual(stats['faculty_count'], 1)
        self.assertEqual(stats['course_count'], 1)
        self.assertEqual(stats['department_name'], "Computer Science")
        self.assertIn('utilization_rate', stats)


class TestSystemIntegration(unittest.TestCase):
    """Test system integration and complex scenarios"""
    
    def setUp(self):
        """Set up a complete university system for integration testing"""
        # This would typically use the main UniversityManagementSystem
        # For now, we'll create a simplified version
        self.reg_system = RegistrationSystem()
        
        # Create departments
        cs_dept = Department("CS", "Computer Science")
        math_dept = Department("MATH", "Mathematics")
        
        # Create courses with prerequisites
        courses_data = [
            ("CS101", "Programming I", [], cs_dept),
            ("CS201", "Programming II", ["CS101"], cs_dept),
            ("CS301", "Algorithms", ["CS201", "MATH201"], cs_dept),
            ("MATH101", "Calculus I", [], math_dept),
            ("MATH201", "Discrete Math", ["MATH101"], math_dept),
        ]
        
        for course_id, name, prereqs, dept in courses_data:
            course = Course(course_id, name, 3, 20)
            for prereq in prereqs:
                course.add_prerequisite(prereq)
            dept.add_course(course)
        
        self.reg_system.add_department(cs_dept)
        self.reg_system.add_department(math_dept)
        
        # Create student progression scenarios
        self.reg_system.update_student_record("S001", [])  # New student
        self.reg_system.update_student_record("S002", ["CS101", "MATH101"])  # Second year
        self.reg_system.update_student_record("S003", ["CS101", "CS201", "MATH101", "MATH201"])  # Advanced
    
    def test_student_progression(self):
        """Test realistic student progression through courses"""
        # New student should be able to take introductory courses
        success, _ = self.reg_system.register_student("S001", "CS101")
        self.assertTrue(success)
        
        success, _ = self.reg_system.register_student("S001", "MATH101")
        self.assertTrue(success)
        
        # But not advanced courses
        success, message = self.reg_system.register_student("S001", "CS301")
        self.assertFalse(success)
        self.assertIn("prerequisite", message.lower())
        
        # Second year student can take intermediate courses
        success, _ = self.reg_system.register_student("S002", "CS201")
        self.assertTrue(success)
        
        success, _ = self.reg_system.register_student("S002", "MATH201")
        self.assertTrue(success)
        
        # Advanced student can take advanced courses
        success, _ = self.reg_system.register_student("S003", "CS301")
        self.assertTrue(success)
    
    def test_course_capacity_management(self):
        """Test course capacity and waitlist management"""
        course = self.reg_system.all_courses["CS101"]
        
        # Fill course to capacity (20 students)
        for i in range(20):
            student_id = f"S{100+i:03d}"
            self.reg_system.update_student_record(student_id, [])
            success, _ = self.reg_system.register_student(student_id, "CS101")
            self.assertTrue(success)
        
        # Next student should go to waitlist
        self.reg_system.update_student_record("S200", [])
        success, message = self.reg_system.register_student("S200", "CS101")
        self.assertFalse(success)
        self.assertIn("waitlist", message.lower())
        
        # Drop a student and verify waitlist promotion
        course.drop_student("S100")
        # Waitlist student should be automatically enrolled
        self.assertIn("S200", course.enrolled_students)


def run_tests():
    """Run all unit tests"""
    # Create test suite
    test_classes = [
        TestPersonHierarchy,
        TestStudentManagement,
        TestSecureStudentRecord,
        TestFacultyPolymorphism,
        TestCourseManagement,
        TestRegistrationSystem,
        TestDepartmentManagement,
        TestSystemIntegration
    ]
    
    suite = unittest.TestSuite()
    
    # Add all test methods from each class
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split()[0] if 'AssertionError:' in traceback else 'Unknown error'}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split()[-1] if traceback else 'Unknown error'}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("University Management System - Unit Tests")
    print("="*70)
    print("Testing all OOP concepts and system functionality...")
    print()
    
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed! The University Management System is working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the system implementation.")
    
    sys.exit(0 if success else 1)