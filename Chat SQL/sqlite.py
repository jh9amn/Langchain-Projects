import sqlite3

## connect to sqlite
connection = sqlite3.connect('students.db')


## create a cursor object to insert records, create tables
cursor = connection.cursor()

## create a table
table_info = """
create table students(name varchar(25), class varchar(25), section varchar(25), marks int)
"""

cursor.execute(table_info)


## Insert records into the table
cursor.execute('''insert into students values('John', 'Data Science', 'A', 85)''')
cursor.execute('''insert into students values('Alice', 'Data Science', 'B', 90)''')
cursor.execute('''insert into students values('Bob', 'Web Development', 'A', 78)''')
cursor.execute('''insert into students values('Charlie', 'Web Development', 'B', 82)''')

## display the records
cursor.execute('''select * from students''')

data = cursor.execute('''select * from students''')
for row in data:
    print(row)
    
## commit the changes to the database
connection.commit()
connection.close()