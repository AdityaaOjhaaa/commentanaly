import os
import shutil
import sqlite3
from datetime import datetime
import glob
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseBackup:
    def __init__(self, db_path, backup_dir='backups', max_backups=5):
        """
        Initialize the backup utility.
        
        Args:
            db_path (str): Path to the SQLite database file
            backup_dir (str): Directory to store backups
            max_backups (int): Maximum number of backup versions to keep
        """
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self):
        """Create a new backup of the database."""
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(
                self.backup_dir, 
                f'users_db_backup_{timestamp}.db'
            )
            
            # Copy the database file
            shutil.copy2(self.db_path, backup_path)
            
            # Verify the backup
            if self._verify_backup(backup_path):
                logger.info(f"Backup created successfully: {backup_path}")
                
                # Clean up old backups
                self._cleanup_old_backups()
                return True
            else:
                logger.error("Backup verification failed")
                os.remove(backup_path)  # Remove corrupted backup
                return False
                
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return False
    
    def _verify_backup(self, backup_path):
        """Verify the integrity of the backup file."""
        try:
            # Try to connect to the backup database
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()
            
            # Check if main tables exist and can be queried
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            # Verify essential tables exist
            essential_tables = {'user', 'analysis'}
            existing_tables = {table[0] for table in tables}
            
            if not essential_tables.issubset(existing_tables):
                logger.error("Backup verification failed: Missing essential tables")
                return False
                
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Backup verification failed: {str(e)}")
            return False
    
    def _cleanup_old_backups(self):
        """Remove old backups keeping only the most recent ones."""
        try:
            # Get list of all backups
            backup_pattern = os.path.join(self.backup_dir, 'users_db_backup_*.db')
            backups = sorted(glob.glob(backup_pattern))
            
            # Remove oldest backups if we have more than max_backups
            while len(backups) > self.max_backups:
                oldest_backup = backups.pop(0)
                os.remove(oldest_backup)
                logger.info(f"Removed old backup: {oldest_backup}")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    def restore_backup(self, backup_path=None):
        """
        Restore the database from a backup.
        If no backup_path is provided, uses the most recent backup.
        """
        try:
            # If no specific backup provided, use most recent
            if backup_path is None:
                backup_pattern = os.path.join(self.backup_dir, 'users_db_backup_*.db')
                backups = sorted(glob.glob(backup_pattern))
                
                if not backups:
                    logger.error("No backups found to restore")
                    return False
                    
                backup_path = backups[-1]
            
            # Verify the backup before restoring
            if not self._verify_backup(backup_path):
                logger.error("Restore failed: Backup verification failed")
                return False
            
            # Create a backup of current database before restoring
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pre_restore_backup = os.path.join(
                self.backup_dir, 
                f'pre_restore_backup_{timestamp}.db'
            )
            shutil.copy2(self.db_path, pre_restore_backup)
            
            # Restore the backup
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return False
    
    def list_backups(self):
        """Return a list of available backups with their timestamps."""
        try:
            backup_pattern = os.path.join(self.backup_dir, 'users_db_backup_*.db')
            backups = sorted(glob.glob(backup_pattern))
            
            backup_info = []
            for backup in backups:
                filename = os.path.basename(backup)
                timestamp = filename.replace('users_db_backup_', '').replace('.db', '')
                size = os.path.getsize(backup) / (1024 * 1024)  # Convert to MB
                
                backup_info.append({
                    'path': backup,
                    'timestamp': timestamp,
                    'size_mb': round(size, 2)
                })
            
            return backup_info
            
        except Exception as e:
            logger.error(f"Failed to list backups: {str(e)}")
            return [] 
