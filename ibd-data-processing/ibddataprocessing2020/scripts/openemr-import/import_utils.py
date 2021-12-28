from sqlalchemy.orm import sessionmaker


def truncate_table(engine, table_name):
    __exec_against_session(engine, f'TRUNCATE TABLE {table_name};')


def delete_where(engine, table_name, where):
    __exec_against_session(engine, f'DELETE FROM {table_name} WHERE {where}')


def __exec_against_session(engine, command):
    Session = sessionmaker(bind=engine)
    session = Session()
    session.execute(command)
    session.commit()
    session.close()
