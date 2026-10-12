import os
import json
import re
from pathlib import Path
from datetime import datetime, date, timedelta

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
DATA_FILE = BASE_DIR / "data.json"

load_dotenv(dotenv_path=ENV_FILE, override=True)

TOKEN = os.getenv("TOKEN")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

client = InferenceClient(
    model=MODEL,
    token=TOKEN
) if TOKEN else None


# ============================================================
# 2. DATA STORAGE
# ============================================================

DEFAULT_DATA = {
    "student": {},
    "attendance": [],
    "assignments": [],
    "exams": [],
    "marks": [],
    "tasks": [],
    "timetable": {},
    "projects": [],
    "reminders": [],
    "notes": []
}


def load_data():
    if not DATA_FILE.exists():
        save_data(DEFAULT_DATA)
        return DEFAULT_DATA.copy()

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        for key, value in DEFAULT_DATA.items():
            if key not in data:
                data[key] = value.copy() if isinstance(value, (dict, list)) else value

        return data

    except (json.JSONDecodeError, OSError):
        print("Warning: data.json could not be read. Starting with default data.")
        return {key: value.copy() if isinstance(value, (dict, list)) else value
                for key, value in DEFAULT_DATA.items()}


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


data = load_data()


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def ask(prompt):
    return input(prompt).strip()


def get_number(prompt, minimum=0):
    while True:
        try:
            value = float(input(prompt).strip())
            if value < minimum:
                print(f"Enter a number greater than or equal to {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def get_date(prompt):
    while True:
        value = ask(prompt)
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            print("Use the date format YYYY-MM-DD.")


def next_id(items):
    return max((item.get("id", 0) for item in items), default=0) + 1


def show_records(title, records):
    print(f"\n===== {title} =====")

    if not records:
        print("No records found.")
        return

    for item in records:
        print(json.dumps(item, indent=2, ensure_ascii=False))


def add_record(section, record):
    record["id"] = next_id(data[section])
    data[section].append(record)
    save_data(data)
    print(f"Record saved successfully. ID: {record['id']}")


def delete_record(section):
    if not data[section]:
        print("No records available.")
        return

    show_records(section.title(), data[section])

    try:
        record_id = int(ask("Enter the record ID to delete: "))
    except ValueError:
        print("Invalid ID.")
        return

    for item in data[section]:
        if item.get("id") == record_id:
            data[section].remove(item)
            save_data(data)
            print("Record deleted successfully.")
            return

    print("Record ID not found.")


# ============================================================
# 4. TIMETABLE ANALYSIS
# ============================================================

def timetable_analysis():
    print("\n===== TIMETABLE ANALYSIS =====")
    print("1. Add or update a timetable entry")
    print("2. View timetable")
    print("3. Analyze a day")

    choice = ask("Choose an option: ")

    if choice == "1":
        day = ask("Enter day (e.g. Monday): ").title()
        subject = ask("Enter subject: ")
        start = ask("Enter start time: ")
        end = ask("Enter end time: ")
        room = ask("Enter room or classroom: ")

        data["timetable"].setdefault(day, [])
        data["timetable"][day].append({
            "subject": subject,
            "start": start,
            "end": end,
            "room": room
        })

        save_data(data)
        print("Timetable entry saved.")

    elif choice == "2":
        if not data["timetable"]:
            print("No timetable entries found.")
        else:
            for day, entries in data["timetable"].items():
                print(f"\n{day}:")
                for entry in entries:
                    print(
                        f"{entry['start']} - {entry['end']} | "
                        f"{entry['subject']} | Room: {entry['room']}"
                    )

    elif choice == "3":
        day = ask("Enter the day to analyze: ").title()
        entries = data["timetable"].get(day, [])

        if not entries:
            print("No timetable available for that day.")
        else:
            print(f"\nClasses on {day}: {len(entries)}")
            for entry in entries:
                print(
                    f"{entry['start']} - {entry['end']}: "
                    f"{entry['subject']} ({entry['room']})"
                )

    else:
        print("Invalid choice.")


# ============================================================
# 5. ATTENDANCE MANAGEMENT
# ============================================================

def attendance_percentage():
    print("\n===== ATTENDANCE PERCENTAGE =====")

    subject = ask("Enter subject name: ")
    attended = get_number("Classes attended: ")
    total = get_number("Total classes held: ")

    if total == 0:
        print("Total classes must be greater than zero.")
        return

    if attended > total:
        print("Classes attended cannot exceed total classes.")
        return

    percentage = attended / total * 100

    print(f"Subject: {subject}")
    print(f"Attendance: {percentage:.2f}%")

    if percentage >= 75:
        print("Status: At or above the 75% attendance threshold.")
    else:
        print("Status: Below 75%. Check your university attendance rules.")

    required = max(0, int((0.75 * total - attended) / 0.25) + 1)

    if percentage < 75:
        print(
            "As a planning estimate, consecutive classes attended "
            "without missing any may help you reach 75%."
        )

    data["attendance"].append({
        "subject": subject,
        "attended": int(attended),
        "total": int(total),
        "percentage": round(percentage, 2),
        "date": str(date.today())
    })

    save_data(data)


def mark_attendance():
    print("\n===== MARK ATTENDANCE =====")

    subject = ask("Enter subject name: ")
    status = ask("Enter status (present/absent): ").lower()

    if status not in ("present", "absent"):
        print("Enter present or absent.")
        return

    existing = next(
        (item for item in data["attendance"]
         if item.get("subject", "").lower() == subject.lower()
         and "status" in item),
        None
    )

    record = {
        "subject": subject,
        "status": status,
        "date": str(date.today())
    }

    if existing:
        record["id"] = existing.get("id", next_id(data["attendance"]))
    else:
        record["id"] = next_id(data["attendance"])

    data["attendance"].append(record)
    save_data(data)
    print(f"Attendance marked: {subject} - {status}")


def attendance_summary():
    records = [item for item in data["attendance"] if "status" in item]

    if not records:
        print("No individual attendance records found.")
        return

    subjects = {}

    for record in records:
        subject = record["subject"]
        subjects.setdefault(subject, {"present": 0, "total": 0})
        subjects[subject]["total"] += 1

        if record["status"] == "present":
            subjects[subject]["present"] += 1

    print("\n===== ATTENDANCE SUMMARY =====")

    for subject, values in subjects.items():
        percentage = values["present"] / values["total"] * 100
        print(
            f"{subject}: {values['present']}/{values['total']} "
            f"classes ({percentage:.2f}%)"
        )


# ============================================================
# 6. ASSIGNMENT MANAGER
# ============================================================

def assignment_manager():
    print("\n===== ASSIGNMENT MANAGER =====")
    print("1. Add assignment")
    print("2. View assignments")
    print("3. Mark assignment completed")
    print("4. Delete assignment")

    choice = ask("Choose an option: ")

    if choice == "1":
        title = ask("Assignment title: ")
        subject = ask("Subject: ")
        deadline = get_date("Deadline (YYYY-MM-DD): ")

        add_record("assignments", {
            "title": title,
            "subject": subject,
            "deadline": deadline,
            "status": "pending"
        })

    elif choice == "2":
        show_records("ASSIGNMENTS", data["assignments"])

    elif choice == "3":
        update_status("assignments")

    elif choice == "4":
        delete_record("assignments")

    else:
        print("Invalid choice.")


def update_status(section):
    if not data[section]:
        print("No records available.")
        return

    show_records(section.title(), data[section])

    try:
        record_id = int(ask("Enter record ID: "))
    except ValueError:
        print("Invalid ID.")
        return

    for item in data[section]:
        if item.get("id") == record_id:
            item["status"] = ask("Enter new status: ").lower()
            save_data(data)
            print("Status updated.")
            return

    print("Record not found.")


# ============================================================
# 7. EXAM PREPARATION PLANNER
# ============================================================

def exam_planner():
    print("\n===== EXAM PREPARATION PLANNER =====")
    print("1. Add exam")
    print("2. View exams")
    print("3. Generate a study plan")

    choice = ask("Choose an option: ")

    if choice == "1":
        subject = ask("Subject: ")
        exam_date = get_date("Exam date (YYYY-MM-DD): ")
        topics = ask("Topics to prepare: ")

        add_record("exams", {
            "subject": subject,
            "date": exam_date,
            "topics": topics,
            "status": "upcoming"
        })

    elif choice == "2":
        show_records("EXAMS", data["exams"])

    elif choice == "3":
        subject = ask("Subject for study plan: ")
        topics = ask("Topics to study: ")
        days = int(get_number("Available study days: ", 1))

        topic_list = [topic.strip() for topic in topics.split(",") if topic.strip()]

        if not topic_list:
            print("Enter at least one topic.")
            return

        print(f"\nStudy plan for {subject}:")

        for index, topic in enumerate(topic_list):
            study_day = index % days + 1
            print(f"Day {study_day}: Study {topic}")

        print("Reserve time for revision and practice questions.")

    else:
        print("Invalid choice.")


# ============================================================
# 8. MARKS AND CGPA CALCULATOR
# ============================================================

def marks_calculator():
    print("\n===== MARKS AND CGPA CALCULATOR =====")
    print("1. Add marks")
    print("2. View marks")
    print("3. Calculate percentage")
    print("4. Calculate CGPA from grade points")

    choice = ask("Choose an option: ")

    if choice == "1":
        subject = ask("Subject: ")
        obtained = get_number("Marks obtained: ")
        maximum = get_number("Maximum marks: ", 1)

        if obtained > maximum:
            print("Obtained marks cannot exceed maximum marks.")
            return

        add_record("marks", {
            "subject": subject,
            "obtained": obtained,
            "maximum": maximum
        })

    elif choice == "2":
        show_records("MARKS", data["marks"])

    elif choice == "3":
        if not data["marks"]:
            print("No marks recorded.")
            return

        obtained = sum(item["obtained"] for item in data["marks"])
        maximum = sum(item["maximum"] for item in data["marks"])

        if maximum == 0:
            print("Maximum marks cannot be zero.")
            return

        print(f"Overall percentage: {obtained / maximum * 100:.2f}%")

    elif choice == "4":
        count = int(get_number("Number of subjects: ", 1))
        points = []
        credits = []

        for index in range(count):
            print(f"Subject {index + 1}")
            points.append(get_number("Grade point: "))
            credits.append(get_number("Credit: ", 1))

        total_credits = sum(credits)
        cgpa = sum(p * c for p, c in zip(points, credits)) / total_credits

        print(f"Weighted CGPA: {cgpa:.2f}")
        print("Confirm your university's grading scale and formula.")

    else:
        print("Invalid choice.")


# ============================================================
# 9. DEADLINE REMINDERS
# ============================================================

def reminder_manager():
    print("\n===== DEADLINE REMINDER =====")
    print("1. Add reminder")
    print("2. View reminders")

    choice = ask("Choose an option: ")

    if choice == "1":
        title = ask("Reminder title: ")
        deadline = get_date("Deadline (YYYY-MM-DD): ")
        details = ask("Details: ")

        add_record("reminders", {
            "title": title,
            "deadline": deadline,
            "details": details
        })

    elif choice == "2":
        show_records("REMINDERS", data["reminders"])

        today = date.today()

        for reminder in data["reminders"]:
            try:
                due = datetime.strptime(
                    reminder["deadline"], "%Y-%m-%d"
                ).date()

                days_left = (due - today).days

                if days_left < 0:
                    print(f"OVERDUE: {reminder['title']}")
                elif days_left == 0:
                    print(f"DUE TODAY: {reminder['title']}")
                else:
                    print(f"{reminder['title']}: {days_left} day(s) left")

            except (ValueError, KeyError):
                continue

    else:
        print("Invalid choice.")


# ============================================================
# 10. DAILY STUDENT PLANNER
# ============================================================

def daily_planner():
    print("\n===== DAILY STUDENT PLANNER =====")
    print("1. Add task")
    print("2. View tasks")
    print("3. Mark task completed")

    choice = ask("Choose an option: ")

    if choice == "1":
        title = ask("Task title: ")
        task_date = get_date("Task date (YYYY-MM-DD): ")
        priority = ask("Priority (low/medium/high): ").lower()

        add_record("tasks", {
            "title": title,
            "date": task_date,
            "priority": priority,
            "status": "pending"
        })

    elif choice == "2":
        show_records("TASKS", data["tasks"])

    elif choice == "3":
        update_status("tasks")

    else:
        print("Invalid choice.")


# ============================================================
# 11. PROJECT AND INTERNSHIP ASSISTANT
# ============================================================

def project_manager():
    print("\n===== PROJECT AND INTERNSHIP ASSISTANT =====")
    print("1. Add project or internship")
    print("2. View projects")
    print("3. Update status")

    choice = ask("Choose an option: ")

    if choice == "1":
        title = ask("Project/internship title: ")
        category = ask("Type (project/internship): ")
        deadline = ask("Deadline (YYYY-MM-DD, or leave blank): ")

        add_record("projects", {
            "title": title,
            "category": category,
            "deadline": deadline,
            "status": "in progress"
        })

    elif choice == "2":
        show_records("PROJECTS AND INTERNSHIPS", data["projects"])

    elif choice == "3":
        update_status("projects")

    else:
        print("Invalid choice.")


# ============================================================
# 12. NOTES AND QUIZ GENERATOR
# ============================================================

def notes_and_quiz():
    print("\n===== NOTES AND QUIZ GENERATOR =====")
    print("1. Save notes")
    print("2. View saved notes")
    print("3. Generate quiz from text using AI")
    print("4. Generate study notes using AI")

    choice = ask("Choose an option: ")

    if choice == "1":
        title = ask("Notes title: ")
        content = ask("Paste your notes: ")

        add_record("notes", {
            "title": title,
            "content": content,
            "date": str(date.today())
        })

    elif choice == "2":
        show_records("NOTES", data["notes"])

    elif choice == "3":
        text_content = ask("Enter topic or study text: ")
        print(ai_response(
            "Create 5 educational quiz questions from the following text. "
            "Include the correct answers after the questions:\n" + text_content
        ))

    elif choice == "4":
        text_content = ask("Enter topic or study text: ")
        print(ai_response(
            "Create clear, structured student study notes from this material:\n"
            + text_content
        ))

    else:
        print("Invalid choice.")


# ============================================================
# 13. HUGGING FACE AI
# ============================================================

def ai_response(prompt):
    if client is None:
        return (
            "Hugging Face token is missing. Check that your .env file "
            "contains TOKEN=your_actual_token."
        )

    try:
        result = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful College AI Agent. "
                        "Help students with academic doubts, programming, "
                        "study planning, assignments, projects, and internships. "
                        "Give clear, accurate, well-structured answers."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.5
        )

        return result.choices[0].message.content or "No response was generated."

    except Exception as error:
        return (
            "The AI request failed. Check your Hugging Face token, model "
            "availability, internet connection, and enabled inference provider.\n"
            f"Details: {error}"
        )


def ai_doubt_solver():
    print("\n===== AI DOUBT SOLVER =====")
    question = ask("Enter your question: ")

    if question:
        print("\nAI Agent:")
        print(ai_response(question))


# ============================================================
# 14. CENTRAL REQUEST HANDLER
# ============================================================

def process_user_request(user_input):
    text = user_input.lower()

    # Attendance
    if any(word in text for word in ["attendance percentage", "calculate attendance"]):
        attendance_percentage()
        return "Attendance calculation completed."

    if "mark attendance" in text:
        mark_attendance()
        return "Attendance operation completed."

    if "attendance summary" in text or "attendance records" in text:
        attendance_summary()
        return "Attendance summary displayed."

    # Timetable
    if any(word in text for word in ["timetable", "class schedule", "class timing"]):
        timetable_analysis()
        return "Timetable operation completed."

    # Assignments
    if any(word in text for word in ["assignment", "homework"]):
        assignment_manager()
        return "Assignment operation completed."

    # Exams
    if any(word in text for word in ["exam planner", "exam preparation", "study plan"]):
        exam_planner()
        return "Exam planning operation completed."

    if "exam" in text:
        exam_planner()
        return "Exam operation completed."

    # Marks
    if any(word in text for word in ["marks", "cgpa", "percentage calculator"]):
        marks_calculator()
        return "Marks operation completed."
    
    # Notes and quizzes
    if any(word in text for word in ["notes", "quiz", "flashcard"]):
        notes_and_quiz()
        return "Notes or quiz operation completed."

    # Reminders
    if any(word in text for word in ["reminder", "deadline"]):
        reminder_manager()
        return "Reminder operation completed."

    # Daily planner
    if any(word in text for word in ["daily planner", "daily task", "my tasks", "to-do"]):
        daily_planner()
        return "Task operation completed."

    # Projects and internships
    if any(word in text for word in ["project", "internship"]):
        project_manager()
        return "Project operation completed."

    # General AI question
    return ai_response(user_input)

# ============================================================
# 15. TERMINAL MENU
# ============================================================

def show_help():
    print("""
================ COLLEGE AI AGENT ================

Available Features:

1.  Timetable Analysis
2.  Attendance Percentage Calculator
3.  Mark Attendance
4.  Assignment Manager
5.  Exam Preparation Planner
6.  AI Doubt Solver
7.  Notes and Quiz Generator
8.  Marks and CGPA Calculator
9.  Deadline Reminder
10. Daily Student Planner
11. Project and Internship Assistant

Commands:
help       - Show this feature list
attendance - View attendance summary
exit       - Close the program

You can also type a question directly, such as:
- Explain Python functions.
- Help me prepare for my exams.
- Create a quiz on data structures.
- Show my timetable.
- Help me plan my assignments.

===================================================
""")

def main():
    print("=" * 50)
    print("             COLLEGE AI AGENT")
    print("=" * 50)

    student_name = data.get("student", {}).get("name", "")

    if not student_name:
        student_name = ask("Enter your name: ") or "Student"
        data["student"] = {"name": student_name}
        save_data(data)

    print(f"\nWelcome, {student_name}!")
    print("Type 'help' to see available features.")
    print("Type 'exit' to close the program.")

    while True:
        try:
            user_input = input(f"\n{student_name}: ").strip()

            if not user_input:
                continue

            command = user_input.lower()

            if command in ("exit", "quit", "bye"):
                print(f"Goodbye, {student_name}. College AI Agent closed.")
                break

            if command == "help":
                show_help()
                continue

            if command in ("attendance", "attendance summary"):
                attendance_summary()
                continue

            response = process_user_request(user_input)
            print(f"\nAI Agent: {response}")

        except KeyboardInterrupt:
            print("\nCollege AI Agent closed.")
            break

        except Exception as error:
            print(f"\nAn error occurred: {error}")
            print("Please try again.")
            
# ============================================================
# 16. START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()