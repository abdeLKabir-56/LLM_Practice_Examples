from z3 import Solver, Int, And, Or, Not, sat


solver = Solver()

def verify_pizza_selection(pizza_name,pizza_size,pizza_category):
    # Define Constraints
    pizza_name_constraint = Or(pizza_name == "Margherita", pizza_name == "Pepperoni", pizza_name == "Veggie")
    pizza_size_constraint = Or(pizza_size == "Small", pizza_size == "Medium", pizza_size == "Large")
    pizza_category_constraint = Or(pizza_category == "Vegetarian", pizza_category == "Non-Vegetarian", pizza_category == "Vegan")
    
    
    #add constraints to the solver
    solver.add(pizza_name_constraint)
    solver.add(pizza_size_constraint)
    solver.add(pizza_category_constraint)
    
    if solver.check() == sat:
        return True
    return False