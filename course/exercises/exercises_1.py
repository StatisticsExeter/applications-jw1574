def sum_list(numbers):
    """Return the sum of all elements in the list 'numbers'."""
    return sum(numbers)
sum_list([1,5,6])

def first_of_tuple(t):
    """Return the first element of the tuple 't'."""
    return t[0]
first_of_tuple((100,70,60))

def has_key(d, key):
    """Return True if 'key' exists in dictionary 'd', else False."""
    return key in d 
my_dict={"joe","greggs","dog"}
has_key(my_dict,"devon")

def round_float(f):
    """Round the float 'f' to 2 decimal places."""
    return round(f,2)
round_float(4.88745347856)

def reverse_list(lst):
    """Return a new list that is the reverse of 'lst'."""
    return lst[::-1]
reverse_list([8,7,5,4,6,3,0])

def count_occurrences(lst, item):
    """For a list of items 'lst', count how many times element 'item' occurs."""
    return lst.count(item)
count_occurrences([3,3,3,5,7,8,6,4,56,89,0],67)

def tuples_to_dict(pairs):
    """Convert a list of (key, value) tuples 'pairs' into a dictionary."""
    return dict(pairs)
tuples_to_dict([("name", "Alice"), ("age", 25), ("city", "Paris")])


def string_length(s):
    """Return the number of characters in string 's'."""
    return len(s)
string_length("Joe is so cool")

def unique_elements(lst):
    """Return a list of unique elements from 'lst'."""
    return list(set(lst))
unique_elements([8,7,6,9,4,23,6,43,7])

def swap_dict(d):
    """Return a new dictionary with keys and values of 'd' swapped."""
    return {value:key for key, value in d.items()}
swap_dict({1:'a',7:'b'})

