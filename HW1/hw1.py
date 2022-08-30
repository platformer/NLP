import sys
import re
import pickle


class Person:
    def __init__(self, last, first, mi, id, phone):
        self.last = last
        self.first = first
        self.mi = mi
        self.id = id
        self.phone = phone

    def display(self):
        out = f'Employee id: {self.id}\n\t{self.first} {self.mi} {self.last}\n\t{self.phone}'
        print(out)


def process_data(filepath):
    employees = {}

    with open(filepath, 'r') as f:
        f.readline() # skip first line -- column headers
        for line in f:
            fields = line.strip().split(',') # remove whitespace and get attribute values

            # uppercase first letter of names, lowercase rest
            last = fields[0][0].upper() + fields[0][1:].lower()
            first = fields[1][0].upper() + fields[1][1:].lower()

            # uppercase mi if it exists, otherwise enter 'X'
            mi = fields[2].upper() if fields[2] else 'X'

            id = fields[3].upper()
            # keep asking for id entry until it matches correct format
            while not re.match(r'\D{2}\d{4}', id):
                print(f'ERROR: id must be 2 letters followed by 4 numbers: {id}')
                id = input(f'Enter id for {first} {mi} {last}:\n').upper()
                print()
            
            # remove all delimiters, then insert hyphens
            phone = ''.join(re.split(r'[\-\s\.]+', fields[4]))
            phone = phone[:3] + '-' + phone[3:6] + '-' + phone[6:]

            employees[id] = Person(last, first, mi, id, phone)
    
    return employees


def main(argv):
    if len(argv) < 2:
        print('ERROR: must specify relative path to data')
        return
    
    employees = process_data(argv[1])

    # pickling
    dump_file = 'employee_dump.pickle'
    with open(dump_file, 'wb') as f:
        pickle.dump(employees, f)

    # unpickling to test if data was preserved
    with open(dump_file, 'rb') as f:
        employees2 = pickle.load(f)
        print('Employees list:\n')
        for (_, emp) in employees2.items():
            emp.display()
            print()


if __name__ == '__main__':
    main(sys.argv)