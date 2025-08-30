import sqlite3
import pandas as pd
from datetime import datetime
import email
import logging
from typing import List, Dict
from enum import Enum
from dataclasses import dataclass
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FEEDBACK_DB = 'feedback.db'
TABLE_NAME = 'feedback'

# Enum for issue categories
class IssueCategory(Enum):
    """Enum for issue categories"""
    BUG = 'bug'
    FEATURE_REQUEST = 'feature_request'
    PERFORMANCE_ISSUE = 'performance_issue'
    OTHER = 'other'

# Data class for feedback
@dataclass
class Feedback:
    """Data class for feedback"""
    id: int
    user_id: int
    issue_category: IssueCategory
    description: str
    created_at: datetime

# Exception classes
class FeedbackCollectorError(Exception):
    """Base exception class for feedback collector"""
    pass

class FeedbackCollector:
    """Class for collecting and processing user feedback"""
    def __init__(self, db_name: str = FEEDBACK_DB):
        """Initialize feedback collector"""
        self.db_name = db_name
        self.lock = Lock()
        self.create_table()

    def create_table(self):
        """Create feedback table in database"""
        with self.lock:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        id INTEGER PRIMARY KEY,
                        user_id INTEGER,
                        issue_category TEXT,
                        description TEXT,
                        created_at TEXT
                    )
                """)
                conn.commit()

    def store_feedback(self, user_id: int, issue_category: IssueCategory, description: str) -> int:
        """Store feedback in database"""
        with self.lock:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    INSERT INTO {TABLE_NAME} (user_id, issue_category, description, created_at)
                    VALUES (?, ?, ?, ?)
                """, (user_id, issue_category.value, description, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                return cursor.lastrowid

    def categorize_issues(self) -> Dict[IssueCategory, List[Feedback]]:
        """Categorize issues"""
        with self.lock:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {TABLE_NAME}")
                rows = cursor.fetchall()
                issues = {}
                for row in rows:
                    feedback = Feedback(row[0], row[1], IssueCategory(row[2]), row[3], datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S'))
                    if feedback.issue_category not in issues:
                        issues[feedback.issue_category] = []
                    issues[feedback.issue_category].append(feedback)
                return issues

    def prioritize_improvements(self) -> List[Feedback]:
        """Prioritize improvements"""
        issues = self.categorize_issues()
        prioritized_issues = []
        for issue_category in issues:
            prioritized_issues.extend(issues[issue_category])
        return prioritized_issues

    def generate_feedback_report(self) -> str:
        """Generate feedback report"""
        issues = self.categorize_issues()
        report = ""
        for issue_category in issues:
            report += f"Issue Category: {issue_category.value}\n"
            for feedback in issues[issue_category]:
                report += f"  - {feedback.description}\n"
        return report

def main():
    feedback_collector = FeedbackCollector()
    user_id = 1
    issue_category = IssueCategory.BUG
    description = "This is a bug description"
    feedback_id = feedback_collector.store_feedback(user_id, issue_category, description)
    logger.info(f"Feedback stored with id {feedback_id}")
    issues = feedback_collector.categorize_issues()
    logger.info(f"Issues categorized: {issues}")
    prioritized_issues = feedback_collector.prioritize_improvements()
    logger.info(f"Issues prioritized: {prioritized_issues}")
    report = feedback_collector.generate_feedback_report()
    logger.info(f"Feedback report generated: {report}")

if __name__ == "__main__":
    main()