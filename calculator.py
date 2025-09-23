import math

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "❌ Cannot divide by zero."
    return x / y

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def square_root(x):
    if x < 0:
        return "❌ Cannot take square root of a negative number."
    return math.sqrt(x)

def modulo(x, y):
    return x % y

def calculator():
    print("📟 Simple Python Calculator")
    print("Available operations:")
    print(" 1. Add")
    print(" 2. Subtract")
    print(" 3. Multiply")
    print(" 4. Divide")
    print(" 5. Square")
    print(" 6. Cube")
    print(" 7. Square Root")
    print(" 8. Modulo")

    choice = input("Select operation (1-8): ")

    if choice in ['1', '2', '3', '4', '8']:
        x = float(input("Enter first number: "))
        y = float(input("Enter second number: "))

        if choice == '1':
            print("Result:", add(x, y))
        elif choice == '2':
            print("Result:", subtract(x, y))
        elif choice == '3':
            print("Result:", multiply(x, y))
        elif choice == '4':
            print("Result:", divide(x, y))
        elif choice == '8':
            print("Result:", modulo(x, y))

    elif choice in ['5', '6', '7']:
        x = float(input("Enter a number: "))

        if choice == '5':
            print("Result:", square(x))
        elif choice == '6':
            print("Result:", cube(x))
        elif choice == '7':
            print("Result:", square_root(x))
    else:
        print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    calculator()
