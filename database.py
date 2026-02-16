# database.py
import sqlite3
import pandas as pd
import uuid
from datetime import datetime, timedelta
from io import BytesIO
import os

class DatasetStorage:
    def __init__(self, db_path="instance/datasets.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise la base de données"""
        os.makedirs("instance", exist_ok=True)
        
        conn = self.get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                filename TEXT,
                rows_count INTEGER,
                cols_count INTEGER
            )
        """)
        
        # Index pour accélérer les recherches
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON datasets(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON datasets(expires_at)")
        
        conn.commit()
        conn.close()
        
        # Nettoie les vieux datasets au démarrage
        self.cleanup_old_datasets()
    
    def get_connection(self):
        """Retourne une connexion à la base"""
        return sqlite3.connect(self.db_path)
    
    def save_dataset(self, session_id, df, filename=""):
        """
        Sauvegarde un DataFrame dans SQLite
        Retourne l'ID unique du dataset
        """
        dataset_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=24)
        
        # Convertir DataFrame en parquet (meilleure compression)
        buffer = BytesIO()
        df.to_parquet(buffer, compression='snappy', index=False)
        data_bytes = buffer.getvalue()
        
        conn = self.get_connection()
        conn.execute("""
            INSERT INTO datasets 
            (id, session_id, data, expires_at, filename, rows_count, cols_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id, 
            session_id, 
            data_bytes,
            expires_at,
            filename,
            df.shape[0],
            df.shape[1]
        ))
        conn.commit()
        conn.close()
        
        return dataset_id
    
    def load_dataset(self, session_id, dataset_id=None):
        """
        Charge un DataFrame depuis SQLite
        Si dataset_id est None, charge le dernier dataset de la session
        """
        conn = self.get_connection()
        
        if dataset_id:
            cursor = conn.execute(
                "SELECT data FROM datasets WHERE id = ? AND session_id = ?",
                (dataset_id, session_id)
            )
        else:
            # Charge le dernier dataset de la session
            cursor = conn.execute("""
                SELECT data FROM datasets 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            buffer = BytesIO(row[0])
            return pd.read_parquet(buffer)
        return None
    
    def get_dataset_info(self, session_id, dataset_id=None):
        """Récupère les métadonnées d'un dataset"""
        conn = self.get_connection()
        
        if dataset_id:
            cursor = conn.execute("""
                SELECT id, filename, rows_count, cols_count, created_at, expires_at
                FROM datasets 
                WHERE id = ? AND session_id = ?
            """, (dataset_id, session_id))
        else:
            cursor = conn.execute("""
                SELECT id, filename, rows_count, cols_count, created_at, expires_at
                FROM datasets 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'filename': row[1],
                'rows': row[2],
                'cols': row[3],
                'created_at': row[4],
                'expires_at': row[5]
            }
        return None
    
    def cleanup_old_datasets(self, max_hours=24):
        """Supprime les datasets expirés"""
        conn = self.get_connection()
        conn.execute("""
            DELETE FROM datasets 
            WHERE expires_at < datetime('now')
        """)
        deleted = conn.total_changes
        conn.commit()
        conn.close()
        
        if deleted > 0:
            print(f"🗑️  Nettoyage : {deleted} vieux datasets supprimés")
        return deleted
    
    def get_user_datasets(self, session_id):
        """Liste tous les datasets d'un utilisateur"""
        conn = self.get_connection()
        cursor = conn.execute("""
            SELECT id, filename, rows_count, cols_count, created_at
            FROM datasets 
            WHERE session_id = ?
            ORDER BY created_at DESC
        """, (session_id,))
        
        datasets = []
        for row in cursor.fetchall():
            datasets.append({
                'id': row[0],
                'filename': row[1],
                'rows': row[2],
                'cols': row[3],
                'created_at': row[4]
            })
        
        conn.close()
        return datasets
    