import tkinter as tk
from tkinter import messagebox
import math
import numpy as np
from sympy import symbols, solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ScientificCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scientific Calculator")
        self.geometry("600x800")

        self.create_widgets()

    def create_widgets(self):
        self.display = tk.Entry(self, font=('Arial', 20))
        self.display.grid(row=0, column=0, columnspan=8, sticky='nsew')

        self.mode_label = tk.Label(self, text="Mode: DEC", font=('Arial', 10))
        self.mode_label.grid(row=1, column=0, columnspan=8, sticky='nsew')

        # Buttons
        buttons = [
            ('7', 2, 0), ('8', 2, 1), ('9', 2, 2), ('/', 2, 3),
            ('4', 3, 0), ('5', 3, 1), ('6', 3, 2), ('*', 3, 3),
            ('1', 4, 0), ('2', 4, 1), ('3', 4, 2), ('-', 4, 3),
            ('0', 5, 0), ('.', 5, 1), ('%', 5, 2), ('+', 5, 3),
            ('AC', 6, 0), ('π', 6, 1), ('e', 6, 2), ('=', 6, 3)
        ]

        for (text, row, column) in buttons:
            button = tk.Button(self, text=text, font=('Arial', 20), command=lambda t=text: self.on_button_click(t))
            button.grid(row=row, column=column, sticky='nsew')

        additional_buttons = [
            ('√', 2, 4), ('**', 2, 5), ('log', 2, 6), ('sin', 2, 7), ('Plot 2D', 2, 8), ('Plot 3D', 2, 9),
            ('cos', 3, 4), ('tan', 3, 5), ('asin', 3, 6), ('acos', 3, 7),
            ('atan', 4, 4), ('exp', 4, 5), ('abs', 4, 6), ('!', 4, 7),
            ('C', 5, 4), ('P', 5, 5), ('1/x', 5, 6),('Solve Complex', 5, 7),('j', 5, 8),
            ('(', 6, 4), (')', 6, 5),('[', 6, 6), (']', 6, 7),
            ('Transpose', 7, 4), ('Inverse', 7, 5), ('Determinant', 7, 6), ('Adjoint', 7, 7),
            ('Solve Poly', 8, 4), ('Solve Simul', 8, 5),
            ('Mean', 9, 4),('Median', 9, 5),('Mode', 9, 6),('Range', 9, 7),
            ('Q1', 10, 4),('Q3', 10, 5),('Quartile Range', 10, 6),('Semi-Quartile Range', 10, 7),
            ('&', 11, 4), ('|', 11, 5), ('^', 11, 6), ('<<', 11, 7), ('>>', 12, 4),
            ('BIN', 12, 5), ('HEX', 12, 6), ('DEC', 12, 7),
            ('c', 13, 4), ('h', 13, 5), ('Na', 13, 6), ('G', 13, 7),
            ('m to km', 14, 4), ('km to m', 14, 5), ('kg to g', 14, 6), ('g to kg', 14, 7),
            ('°C to °F', 15, 4), ('°F to °C', 15, 5)
        ]

        for (text, row, column) in additional_buttons:
            button = tk.Button(self, text=text, font=('Arial', 15), command=lambda t=text: self.on_button_click(t))
            button.grid(row=row, column=column, sticky='nsew')


    def on_button_click(self, char):
        if char == '=':
            try:
                result = self.evaluate_expression(self.display.get())
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, str(result))
            except Exception as e:
                messagebox.showerror("Error", str(e))
        elif char == 'AC':
            self.display.delete(0, tk.END)
        elif char == '1/x':
            result = self.evaluate_expression(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(1/result))
        elif char == '%':
            result = self.evaluate_expression(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result/100.0))
        elif char == 'Transpose':
            result = self.evaluate_matrix_operation(self.display.get(), 'transpose')
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Inverse':
            result = self.evaluate_matrix_operation(self.display.get(), 'inverse')
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Determinant':
            result = self.evaluate_matrix_operation(self.display.get(), 'determinant')
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Adjoint':
            result = self.evaluate_matrix_operation(self.display.get(), 'adjoint')
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Solve Poly':
            result = self.solve_polynomial(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Solve Simul':
            result = self.solve_simultaneous_equations(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Mean':
            result = self.calculate_mean(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Median':
            result = self.calculate_median(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Mode':
            result = self.calculate_mode(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Range':
            result = self.calculate_range(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Q1':
            result = self.calculate_quartile(self.display.get(), 1)
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Q3':
            result = self.calculate_quartile(self.display.get(), 3)
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Quartile Range':
            q1 = self.calculate_quartile(self.display.get(), 1)
            q3 = self.calculate_quartile(self.display.get(), 3)
            result = q3 - q1
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Semi-Quartile Range':
            q1 = self.calculate_quartile(self.display.get(), 1)
            q3 = self.calculate_quartile(self.display.get(), 3)
            result = (q3 - q1) / 2
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char in ['BIN', 'HEX', 'DEC']:
            result = self.convert_number_system(self.display.get(), char)
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char in ['m to km', 'km to m', 'kg to g', 'g to kg', '°C to °F', '°F to °C']:
            result = self.convert_metric(char, self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, str(result))
        elif char == 'Plot 2D':
            expression = self.display.get()
            # Parse the expression to extract the mathematical function
            eq_parts = expression.split("=")
            if len(eq_parts) != 2:
                messagebox.showerror("Error", "Invalid expression for 2D plotting")
                return
            func_str = eq_parts[1].strip()
            
            # Define the range of x values
            x_values = np.linspace(-10, 10, 400)
            
            # Evaluate the function for each x value
            try:
                y_values = eval(func_str, {'x': x_values})
            except Exception as e:
                messagebox.showerror("Error", f"Invalid expression: {str(e)}")
                return
    
            # Plot the graph
            plt.plot(x_values, y_values)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('2D Plot')
            plt.grid(True)
            plt.show()
        elif char == 'Plot 3D':
            expression = self.display.get()
            eq_parts = expression.split("=")
            if len(eq_parts) != 2:
                messagebox.showerror("Error", "Invalid expression for 3D plotting")
                return
            func_str = eq_parts[1].strip()

            # Define the range of x and y values
            x_values = np.linspace(-10, 10, 400)
            y_values = np.linspace(-10, 10, 400)
            X, Y = np.meshgrid(x_values, y_values)

            # Evaluate the function for each combination of x and y values
            try:
                Z = eval(func_str, {'x': X, 'y': Y})
            except Exception as e:
                messagebox.showerror("Error", f"Invalid expression: {str(e)}")
                return

            # Plot the 3D graph
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Plot')
            plt.show()
        elif char == 'Solve Complex':
            expression = self.display.get()
            try:
                result = self.solve_complex(expression)
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, str(result))
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            self.display.insert(tk.END, char)
        

    def solve_complex(self, expression):
        try:
            # Parse the expression and split it into real and imaginary parts
            parts = expression.split("+")
            if len(parts) != 2:
                raise ValueError("Invalid complex number expression")

            real_part = parts[0].strip()
            imag_part = parts[1].replace("j", "").strip()

            # Convert real and imaginary parts to float
            real = float(real_part)
            imag = float(imag_part)

            # Create a complex number object and return it
            complex_num = complex(real, imag)
            return complex_num
        except Exception as e:
            raise ValueError("Invalid complex number expression")
    def evaluate_expression(self, expression):
        try:
            expression = expression.replace('π', str(math.pi))
            expression = expression.replace('e', str(math.e))
            expression = expression.replace('√', "sqrt")
            expression = expression.replace('exp', str(math.e)+"**")
            expression = expression.replace('c', str(299792458))
            expression = expression.replace('h', str(6.62607015e-34))
            expression = expression.replace('Na', str(6.022e-23))
            expression = expression.replace('G', str(6.67e-11))
            if 'C' in expression:
                n, _, k = expression.partition('C')
                result = math.comb(int(n), int(k))
            elif 'P' in expression:
                n, _, k = expression.partition('P')
                result = math.perm(int(n), int(k))
            else:
                result = eval(expression, {'__builtins__': None, 'j': 1j}, {'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                                                                'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
                                                                'log': math.log10, 'ln': math.log,'sqrt': math.sqrt,})
            return result
        except Exception as e:
            raise ValueError("Invalid expression")

    def evaluate_matrix_operation(self, expression, operation):
        try:
            matrix = np.array(eval(expression))
            if operation == 'transpose':
                result = np.transpose(matrix)
            elif operation == 'inverse':
                result = np.linalg.inv(matrix)
            elif operation == 'determinant':
                result = np.linalg.det(matrix)
            elif operation == 'adjoint':
                result = np.linalg.inv(matrix) * np.linalg.det(matrix)
            else:
                raise ValueError("Invalid matrix operation")
            return result
        except Exception as e:
            raise ValueError("Invalid matrix expression")

    def solve_polynomial(self, expression):
        try:
            expression = expression.replace('π', str(math.pi))
            expression = expression.replace('e', str(math.e))
            expression = expression.replace('^', "**")
            expression = expression.replace('√', "sqrt")
            expression = expression.replace('exp', str(math.e)+"**")
            x = symbols('x')
            equation = eval(expression)
            solution = solve(equation, x)
            return solution
        except Exception as e:
            raise ValueError("Invalid polynomial equation")

    def solve_simultaneous_equations(self, expression):
        try:
            expression = expression.replace('π', str(math.pi))
            expression = expression.replace('e', str(math.e))
            expression = expression.replace('^', "**")
            expression = expression.replace('√', "sqrt")
            expression = expression.replace('exp', str(math.e)+"**")
            equations = expression.split(',')
            print(equations)
            solutions = solve(equations)
            return solutions
        except Exception as e:
            raise ValueError("Invalid simultaneous equations")

    def calculate_mean(self, expression):
        try:
            numbers = eval(expression)
            return sum(numbers) / len(numbers)
        except Exception as e:
            raise ValueError("Invalid input for mean calculation")

    def calculate_median(self, expression):
        try:
            numbers = eval(expression)
            sorted_numbers = sorted(numbers)
            n = len(sorted_numbers)
            if n % 2 == 0:
                return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
            else:
                return sorted_numbers[n//2]
        except Exception as e:
            raise ValueError("Invalid input for median calculation")

    def calculate_mode(self, expression):
        try:
            numbers = eval(expression)
            freq = {}
            for num in numbers:
                if num in freq:
                    freq[num] += 1
                else:
                    freq[num] = 1
            max_freq = max(freq.values())
            modes = [num for num, f in freq.items() if f == max_freq]
            return modes if len(modes) > 1 else modes[0]
        except Exception as e:
            raise ValueError("Invalid input for mode calculation")

    def calculate_range(self, expression):
        try:
            numbers = eval(expression)
            return max(numbers) - min(numbers)
        except Exception as e:
            raise ValueError("Invalid input for range calculation")

    def calculate_quartile(self, expression, quartile_num):
        try:
            numbers = eval(expression)
            sorted_numbers = sorted(numbers)
            n = len(sorted_numbers)
            k = quartile_num * (n + 1) / 4
            if k.is_integer():
                return sorted_numbers[int(k) - 1]
            else:
                i = int(k)
                return sorted_numbers[i - 1] + (k - i) * (sorted_numbers[i] - sorted_numbers[i - 1])
        except Exception as e:
            raise ValueError("Invalid input for quartile calculation")

    def convert_number_system(self, expression, system):
        try:
            if system == 'BIN':
                return bin(int(expression))
            elif system == 'HEX':
                return hex(int(expression))
            elif system == 'DEC':
                return str(int(expression, 2))
        except Exception as e:
            raise ValueError("Invalid input for number system conversion")

    def convert_metric(self, conversion, value):
        try:
            if conversion == 'm to km':
                return float(value) / 1000
            elif conversion == 'km to m':
                return float(value) * 1000
            elif conversion == 'kg to g':
                return float(value) * 1000
            elif conversion == 'g to kg':
                return float(value) / 1000
            elif conversion == '°C to °F':
                return (float(value) * 9/5) + 32
            elif conversion == '°F to °C':
                return (float(value) - 32) * 5/9
        except Exception as e:
            raise ValueError("Invalid input for metric conversion")

if __name__ == "__main__":
    app = ScientificCalculator()
    app.mainloop()
