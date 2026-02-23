import oracledb

oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_20")

def get_connection():
    return oracledb.connect(
        user="system",
        password="412007",
        dsn="localhost:1521/XE"
    )