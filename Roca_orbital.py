#!/usr/bin/env python3
"""
ROCA Orbital Memory System - Standalone Knowledge Visualization

A standalone implementation of the ROCA (Relevance, Orbital, Capsule, Analysis)
knowledge system with orbital visualization capabilities.

This module provides:
- Capsule-based knowledge representation
- Orbital visualization of knowledge capsules
- GPU-accelerated similarity calculations
- Memory operations (add, merge, orbit updates)
- Pygame-based visualization

Usage:
    from roca_orbital import RocaOrbitalMemory

    # Create memory system
    memory = RocaOrbitalMemory()

    # Add knowledge capsules
    memory.add_capsule("Newton's Laws of Motion", character="Newton")
    memory.add_capsule("Einstein's Relativity", character="Einstein")

    # Visualize
    memory.visualize()
"""

import re
import string
import random
import uuid
import time
import math
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID
import numpy as np
import torch
# import pygame  # Temporarily commented for testing
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# GPU and CuPy detection
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# GPU Detection
try:
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_TYPE = "NVIDIA"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        GPU_AVAILABLE = True
        GPU_TYPE = "APPLE"
    else:
        GPU_AVAILABLE = False
        GPU_TYPE = "CPU"
except:
    GPU_AVAILABLE = False
    GPU_TYPE = "CPU"

# ===== HARD-CODED MATHEMATICAL ENGINE =====

class IntegerArithmetic:
    """Exact integer arithmetic implementation"""
    
    @staticmethod
    def add(a: int, b: int) -> int:
        """Add two integers using manual digit-by-digit addition"""
        if a >= 0 and b >= 0:
            return IntegerArithmetic._add_positive(a, b)
        elif a < 0 and b < 0:
            return -IntegerArithmetic._add_positive(-a, -b)
        else:
            # One positive, one negative
            if abs(a) > abs(b):
                return IntegerArithmetic._subtract_positive(abs(a), abs(b)) if a > 0 else -IntegerArithmetic._subtract_positive(abs(a), abs(b))
            else:
                return IntegerArithmetic._subtract_positive(abs(b), abs(a)) if b > 0 else -IntegerArithmetic._subtract_positive(abs(b), abs(a))
    
    @staticmethod
    def _add_positive(a: int, b: int) -> int:
        """Add two positive integers"""
        result = 0
        carry = 0
        place = 1
        
        while a > 0 or b > 0 or carry > 0:
            digit_a = a % 10
            digit_b = b % 10
            sum_digits = digit_a + digit_b + carry
            
            result += (sum_digits % 10) * place
            carry = sum_digits // 10
            
            a //= 10
            b //= 10
            place *= 10
            
        return result
    
    @staticmethod
    def _subtract_positive(a: int, b: int) -> int:
        """Subtract b from a (both positive, a >= b)"""
        result = 0
        borrow = 0
        place = 1
        
        while a > 0 or b > 0:
            digit_a = a % 10
            digit_b = b % 10
            
            diff = digit_a - digit_b - borrow
            if diff < 0:
                diff += 10
                borrow = 1
            else:
                borrow = 0
                
            result += diff * place
            
            a //= 10
            b //= 10
            place *= 10
            
        return result
    
    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiply two integers using manual multiplication"""
        if a == 0 or b == 0:
            return 0
            
        # Handle signs
        sign = 1
        if a < 0:
            sign = -sign
            a = -a
        if b < 0:
            sign = -sign
            b = -b
            
        result = IntegerArithmetic._multiply_positive(a, b)
        return sign * result
    
    @staticmethod
    def _multiply_positive(a: int, b: int) -> int:
        """Multiply two positive integers"""
        result = 0
        place = 1
        
        while b > 0:
            digit = b % 10
            if digit > 0:
                # Add a * digit * place to result
                for _ in range(digit):
                    result = IntegerArithmetic._add_positive(result, a * place)
            place *= 10
            b //= 10
            
        return result


class Rational:
    """Rational number implementation with exact arithmetic"""
    
    def __init__(self, numerator: int, denominator: int = 1):
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
            
        # Normalize sign to numerator
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
            
        # Reduce fraction
        gcd = self._gcd(abs(numerator), denominator)
        self.numerator = numerator // gcd
        self.denominator = denominator // gcd
    
    @staticmethod
    def _gcd(a: int, b: int) -> int:
        """Greatest common divisor using Euclidean algorithm"""
        while b != 0:
            a, b = b, a % b
        return a
    
    def add(self, other: 'Rational') -> 'Rational':
        """Add two rational numbers"""
        new_num = (self.numerator * other.denominator + 
                  other.numerator * self.denominator)
        new_den = self.denominator * other.denominator
        return Rational(new_num, new_den)
    
    def multiply(self, other: 'Rational') -> 'Rational':
        """Multiply two rational numbers"""
        new_num = self.numerator * other.numerator
        new_den = self.denominator * other.denominator
        return Rational(new_num, new_den)
    
    def to_float(self) -> float:
        """Convert to floating point (for display)"""
        return self.numerator / self.denominator
    
    def __str__(self):
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"


class FloatWithError:
    """Floating point with error awareness"""
    
    def __init__(self, value: float, error: float = 0.0):
        self.value = value
        self.error = error
    
    def add(self, other: 'FloatWithError') -> 'FloatWithError':
        """Add with error propagation"""
        result_value = self.value + other.value
        result_error = abs(self.error) + abs(other.error)  # Simple error addition
        return FloatWithError(result_value, result_error)
    
    def multiply(self, other: 'FloatWithError') -> 'FloatWithError':
        """Multiply with error propagation"""
        result_value = self.value * other.value
        # Relative error propagation: δ(z) = |δ(x)| + |δ(y)|
        rel_error = abs(self.error / self.value if self.value != 0 else 0) + \
                   abs(other.error / other.value if other.value != 0 else 0)
        result_error = abs(result_value) * rel_error
        return FloatWithError(result_value, result_error)
    
    def __str__(self):
        if self.error == 0:
            return f"{self.value}"
        return f"{self.value} ± {self.error}"


# ===== END HARD-CODED MATHEMATICAL ENGINE =====

# ===== HARD-CODED ALGEBRAIC ENGINE =====

class AlgebraicExpression:
    """Basic algebraic expression representation"""
    
    def __init__(self, expr_type: str, **kwargs):
        self.expr_type = expr_type  # 'constant', 'variable', 'sum', 'product', 'power', 'fraction'
        if expr_type == 'constant':
            self.value = kwargs['value']  # Rational or int
        elif expr_type == 'variable':
            self.name = kwargs['name']  # string like 'x', 'y'
        elif expr_type == 'sum':
            self.terms = kwargs['terms']  # list of AlgebraicExpression
        elif expr_type == 'product':
            self.factors = kwargs['factors']  # list of AlgebraicExpression
        elif expr_type == 'power':
            self.base = kwargs['base']  # AlgebraicExpression
            self.exponent = kwargs['exponent']  # AlgebraicExpression
        elif expr_type == 'fraction':
            self.numerator = kwargs['numerator']  # AlgebraicExpression
            self.denominator = kwargs['denominator']  # AlgebraicExpression
    
    def __str__(self):
        if self.expr_type == 'constant':
            return str(self.value)
        elif self.expr_type == 'variable':
            return self.name
        elif self.expr_type == 'sum':
            return ' + '.join(str(term) for term in self.terms)
        elif self.expr_type == 'product':
            return '*'.join(str(factor) for factor in self.factors)
        elif self.expr_type == 'power':
            return f'({self.base})^{self.exponent}'
        elif self.expr_type == 'fraction':
            return f'({self.numerator})/({self.denominator})'
        return '?'
    
    def equals(self, other: 'AlgebraicExpression') -> bool:
        """Check if two expressions are structurally equal"""
        if self.expr_type != other.expr_type:
            return False
        
        if self.expr_type == 'constant':
            return self.value == other.value
        elif self.expr_type == 'variable':
            return self.name == other.name
        elif self.expr_type == 'sum':
            if len(self.terms) != len(other.terms):
                return False
            # Sort terms for comparison (commutative)
            self_sorted = sorted(self.terms, key=str)
            other_sorted = sorted(other.terms, key=str)
            return all(a.equals(b) for a, b in zip(self_sorted, other_sorted))
        elif self.expr_type == 'product':
            if len(self.factors) != len(other.factors):
                return False
            # Sort factors for comparison (commutative)
            self_sorted = sorted(self.factors, key=str)
            other_sorted = sorted(other.factors, key=str)
            return all(a.equals(b) for a, b in zip(self_sorted, other_sorted))
        elif self.expr_type == 'power':
            return self.base.equals(other.base) and self.exponent.equals(other.exponent)
        elif self.expr_type == 'fraction':
            return (self.numerator.equals(other.numerator) and 
                   self.denominator.equals(other.denominator))
        return False


class AlgebraicManipulator:
    """Hard-coded algebraic manipulation operations"""
    
    @staticmethod
    def simplify(expr: AlgebraicExpression) -> AlgebraicExpression:
        """Simplify an algebraic expression"""
        if expr.expr_type == 'constant':
            return expr
        elif expr.expr_type == 'variable':
            return expr
        elif expr.expr_type == 'sum':
            # Combine like terms
            simplified_terms = []
            constants_sum = Rational(0, 1)
            variables = {}  # var_name -> coefficient
            
            for term in expr.terms:
                if term.expr_type == 'constant':
                    constants_sum = constants_sum.add(Rational(term.value.numerator, term.value.denominator) 
                                                    if hasattr(term.value, 'numerator') 
                                                    else Rational(term.value, 1))
                elif term.expr_type == 'variable':
                    var_name = term.name
                    coeff = Rational(1, 1)
                    if var_name in variables:
                        variables[var_name] = variables[var_name].add(coeff)
                    else:
                        variables[var_name] = coeff
                elif term.expr_type == 'product':
                    # Check if it's a constant times variable
                    if len(term.factors) == 2:
                        const_factor = None
                        var_factor = None
                        for factor in term.factors:
                            if factor.expr_type == 'constant':
                                const_factor = factor
                            elif factor.expr_type == 'variable':
                                var_factor = factor
                        
                        if const_factor and var_factor:
                            coeff = Rational(const_factor.value.numerator, const_factor.value.denominator) \
                                  if hasattr(const_factor.value, 'numerator') \
                                  else Rational(const_factor.value, 1)
                            var_name = var_factor.name
                            if var_name in variables:
                                variables[var_name] = variables[var_name].add(coeff)
                            else:
                                variables[var_name] = coeff
                        else:
                            simplified_terms.append(AlgebraicManipulator.simplify(term))
                    else:
                        simplified_terms.append(AlgebraicManipulator.simplify(term))
                else:
                    simplified_terms.append(AlgebraicManipulator.simplify(term))
            
            # Reconstruct expression
            final_terms = []
            if constants_sum.numerator != 0:
                final_terms.append(AlgebraicExpression('constant', value=constants_sum))
            
            for var_name, coeff in variables.items():
                if coeff.equals(Rational(1, 1)):
                    final_terms.append(AlgebraicExpression('variable', name=var_name))
                elif coeff.equals(Rational(-1, 1)):
                    # This would need a negative term representation
                    final_terms.append(AlgebraicExpression('product', 
                                                         factors=[AlgebraicExpression('constant', value=Rational(-1, 1)),
                                                                 AlgebraicExpression('variable', name=var_name)]))
                else:
                    final_terms.append(AlgebraicExpression('product',
                                                         factors=[AlgebraicExpression('constant', value=coeff),
                                                                 AlgebraicExpression('variable', name=var_name)]))
            
            final_terms.extend(simplified_terms)
            
            if len(final_terms) == 1:
                return final_terms[0]
            elif len(final_terms) == 0:
                return AlgebraicExpression('constant', value=Rational(0, 1))
            else:
                return AlgebraicExpression('sum', terms=final_terms)
                
        elif expr.expr_type == 'product':
            # Simplify each factor and multiply constants
            simplified_factors = [AlgebraicManipulator.simplify(factor) for factor in expr.factors]
            constants_product = Rational(1, 1)
            non_constant_factors = []
            
            for factor in simplified_factors:
                if factor.expr_type == 'constant':
                    constants_product = constants_product.multiply(
                        Rational(factor.value.numerator, factor.value.denominator) 
                        if hasattr(factor.value, 'numerator') 
                        else Rational(factor.value, 1)
                    )
                else:
                    non_constant_factors.append(factor)
            
            if constants_product.equals(Rational(0, 1)):
                return AlgebraicExpression('constant', value=Rational(0, 1))
            elif constants_product.equals(Rational(1, 1)) and len(non_constant_factors) == 1:
                return non_constant_factors[0]
            elif len(non_constant_factors) == 0:
                return AlgebraicExpression('constant', value=constants_product)
            else:
                final_factors = []
                if not constants_product.equals(Rational(1, 1)):
                    final_factors.append(AlgebraicExpression('constant', value=constants_product))
                final_factors.extend(non_constant_factors)
                return AlgebraicExpression('product', factors=final_factors)
                
        elif expr.expr_type == 'power':
            base_simplified = AlgebraicManipulator.simplify(expr.base)
            exponent_simplified = AlgebraicManipulator.simplify(expr.exponent)
            
            # Special cases
            if exponent_simplified.expr_type == 'constant':
                exp_val = exponent_simplified.value
                if exp_val.equals(Rational(0, 1)):
                    return AlgebraicExpression('constant', value=Rational(1, 1))
                elif exp_val.equals(Rational(1, 1)):
                    return base_simplified
            
            return AlgebraicExpression('power', base=base_simplified, exponent=exponent_simplified)
            
        elif expr.expr_type == 'fraction':
            num_simplified = AlgebraicManipulator.simplify(expr.numerator)
            den_simplified = AlgebraicManipulator.simplify(expr.denominator)
            
            # If both numerator and denominator are constants
            if (num_simplified.expr_type == 'constant' and 
                den_simplified.expr_type == 'constant'):
                num_rat = Rational(num_simplified.value.numerator, num_simplified.value.denominator) \
                         if hasattr(num_simplified.value, 'numerator') \
                         else Rational(num_simplified.value, 1)
                den_rat = Rational(den_simplified.value.numerator, den_simplified.value.denominator) \
                         if hasattr(den_simplified.value, 'numerator') \
                         else Rational(den_simplified.value, 1)
                result_rat = Rational(num_rat.numerator * den_rat.denominator, 
                                    num_rat.denominator * den_rat.numerator)
                return AlgebraicExpression('constant', value=result_rat)
            
            return AlgebraicExpression('fraction', numerator=num_simplified, denominator=den_simplified)
        
        return expr
    
    @staticmethod
    def expand(expr: AlgebraicExpression) -> AlgebraicExpression:
        """Expand algebraic expressions (e.g., (x+1)^2 = x^2 + 2x + 1)"""
        if expr.expr_type == 'power':
            # Handle (a+b)^n for small integer n
            if (expr.base.expr_type == 'sum' and 
                expr.exponent.expr_type == 'constant'):
                exp_val = expr.exponent.value
                if exp_val.equals(Rational(2, 1)):  # (a+b)^2 = a^2 + 2ab + b^2
                    terms = expr.base.terms
                    if len(terms) == 2:
                        a, b = terms
                        a_squared = AlgebraicExpression('power', base=a, exponent=AlgebraicExpression('constant', value=Rational(2, 1)))
                        b_squared = AlgebraicExpression('power', base=b, exponent=AlgebraicExpression('constant', value=Rational(2, 1)))
                        two_ab = AlgebraicExpression('product', 
                                                   factors=[AlgebraicExpression('constant', value=Rational(2, 1)),
                                                           a, b])
                        return AlgebraicExpression('sum', terms=[a_squared, two_ab, b_squared])
        
        elif expr.expr_type == 'product':
            # Expand products of sums using distributive property
            # For now, handle simple cases like (x+1)(x+2)
            if len(expr.factors) == 2:
                f1, f2 = expr.factors
                if f1.expr_type == 'sum' and f2.expr_type == 'sum':
                    # (a+b)(c+d) = ac + ad + bc + bd
                    expanded_terms = []
                    for term1 in f1.terms:
                        for term2 in f2.terms:
                            expanded_terms.append(AlgebraicExpression('product', factors=[term1, term2]))
                    return AlgebraicExpression('sum', terms=expanded_terms)
        
        # If no expansion needed, return as-is
        return expr
    
    @staticmethod
    def factor(expr: AlgebraicExpression) -> AlgebraicExpression:
        """Factor algebraic expressions (e.g., x^2 + 2x + 1 = (x+1)^2)"""
        if expr.expr_type == 'sum':
            # Look for common patterns that can be factored
            # For now, implement basic factoring like x^2 + 2x + 1 = (x+1)^2
            if len(expr.terms) == 3:
                # Check if it's a perfect square trinomial: ax^2 + bx + c
                x_squared_term = None
                x_term = None
                constant_term = None
                
                for term in expr.terms:
                    if term.expr_type == 'power' and term.exponent.expr_type == 'constant':
                        if term.exponent.value.equals(Rational(2, 1)) and term.base.expr_type == 'variable':
                            x_squared_term = term
                    elif term.expr_type == 'variable':
                        x_term = term
                    elif term.expr_type == 'constant':
                        constant_term = term
                
                if x_squared_term and x_term and constant_term:
                    # Check if coefficients form a perfect square
                    # For x^2 + 2x + 1 = (x+1)^2
                    if (x_squared_term.base.name == x_term.name and 
                        constant_term.value.equals(Rational(1, 1))):
                        # This is (x+1)^2
                        return AlgebraicExpression('power',
                                                 base=AlgebraicExpression('sum',
                                                                        terms=[AlgebraicExpression('variable', name=x_term.name),
                                                                              AlgebraicExpression('constant', value=Rational(1, 1))]),
                                                 exponent=AlgebraicExpression('constant', value=Rational(2, 1)))
        
        # If no factoring possible, return as-is
        return expr
    
    @staticmethod
    def substitute(expr: AlgebraicExpression, var_name: str, replacement: AlgebraicExpression) -> AlgebraicExpression:
        """Substitute a variable with an expression"""
        if expr.expr_type == 'constant':
            return expr
        elif expr.expr_type == 'variable':
            if expr.name == var_name:
                return replacement
            else:
                return expr
        elif expr.expr_type == 'sum':
            new_terms = [AlgebraicManipulator.substitute(term, var_name, replacement) 
                        for term in expr.terms]
            return AlgebraicExpression('sum', terms=new_terms)
        elif expr.expr_type == 'product':
            new_factors = [AlgebraicManipulator.substitute(factor, var_name, replacement) 
                          for factor in expr.factors]
            return AlgebraicExpression('product', factors=new_factors)
        elif expr.expr_type == 'power':
            new_base = AlgebraicManipulator.substitute(expr.base, var_name, replacement)
            new_exponent = AlgebraicManipulator.substitute(expr.exponent, var_name, replacement)
            return AlgebraicExpression('power', base=new_base, exponent=new_exponent)
        elif expr.expr_type == 'fraction':
            new_num = AlgebraicManipulator.substitute(expr.numerator, var_name, replacement)
            new_den = AlgebraicManipulator.substitute(expr.denominator, var_name, replacement)
            return AlgebraicExpression('fraction', numerator=new_num, denominator=new_den)
        
        return expr
    
    @staticmethod
    def check_equality(expr1: AlgebraicExpression, expr2: AlgebraicExpression) -> bool:
        """Check if two expressions are mathematically equal"""
        # First, simplify both expressions
        simplified1 = AlgebraicManipulator.simplify(expr1)
        simplified2 = AlgebraicManipulator.simplify(expr2)
        
        # Check structural equality
        return simplified1.equals(simplified2)


# ===== END HARD-CODED ALGEBRAIC ENGINE =====

# ===== HARD-CODED CALCULUS ENGINE =====

class CalculusEngine:
    """Hard-coded calculus operations - differentiation and integration"""
    
    @staticmethod
    def differentiate(expr: AlgebraicExpression, var_name: str = 'x') -> AlgebraicExpression:
        """Compute symbolic derivative with respect to variable"""
        if expr.expr_type == 'constant':
            # d/dx(c) = 0
            return AlgebraicExpression('constant', value=Rational(0, 1))
        
        elif expr.expr_type == 'variable':
            # d/dx(x) = 1, d/dx(other_var) = 0
            if expr.name == var_name:
                return AlgebraicExpression('constant', value=Rational(1, 1))
            else:
                return AlgebraicExpression('constant', value=Rational(0, 1))
        
        elif expr.expr_type == 'sum':
            # d/dx(f + g) = f' + g'
            differentiated_terms = [CalculusEngine.differentiate(term, var_name) for term in expr.terms]
            return AlgebraicExpression('sum', terms=differentiated_terms)
        
        elif expr.expr_type == 'product':
            # d/dx(f*g) = f'*g + f*g' (product rule)
            if len(expr.factors) == 2:
                f, g = expr.factors
                f_prime = CalculusEngine.differentiate(f, var_name)
                g_prime = CalculusEngine.differentiate(g, var_name)
                
                # f'*g
                term1 = AlgebraicExpression('product', factors=[f_prime, g])
                # f*g'
                term2 = AlgebraicExpression('product', factors=[f, g_prime])
                
                return AlgebraicExpression('sum', terms=[term1, term2])
            else:
                # For more than 2 factors, expand using product rule iteratively
                # This is a simplified implementation
                return AlgebraicExpression('constant', value=Rational(0, 1))  # Placeholder
        
        elif expr.expr_type == 'power':
            # d/dx(x^n) = n*x^(n-1) (power rule)
            if (expr.base.expr_type == 'variable' and expr.base.name == var_name and
                expr.exponent.expr_type == 'constant'):
                
                n = expr.exponent.value
                if hasattr(n, 'numerator'):
                    n_val = n.numerator / n.denominator
                else:
                    n_val = float(n)
                
                # n * x^(n-1)
                coeff = AlgebraicExpression('constant', value=Rational(int(n_val), 1))
                new_exponent = AlgebraicExpression('constant', value=Rational(int(n_val - 1), 1))
                power_term = AlgebraicExpression('power', 
                                               base=AlgebraicExpression('variable', name=var_name),
                                               exponent=new_exponent)
                
                return AlgebraicExpression('product', factors=[coeff, power_term])
            
            # Chain rule for more complex cases
            # d/dx(f^g) = f^g * (g'*ln(f) + g*f'/f)
            # This is complex, so for now return a placeholder
            return AlgebraicExpression('constant', value=Rational(0, 1))  # Placeholder for complex cases
        
        elif expr.expr_type == 'fraction':
            # d/dx(f/g) = (f'*g - f*g')/g^2 (quotient rule)
            f, g = expr.numerator, expr.denominator
            f_prime = CalculusEngine.differentiate(f, var_name)
            g_prime = CalculusEngine.differentiate(g, var_name)
            
            # f'*g
            numerator_term1 = AlgebraicExpression('product', factors=[f_prime, g])
            # f*g'
            numerator_term2 = AlgebraicExpression('product', factors=[f, g_prime])
            # f'*g - f*g'
            numerator = AlgebraicExpression('sum', terms=[numerator_term1, 
                                                         AlgebraicExpression('product', 
                                                                           factors=[AlgebraicExpression('constant', value=Rational(-1, 1)),
                                                                                   numerator_term2])])
            # g^2
            denominator = AlgebraicExpression('power', base=g, exponent=AlgebraicExpression('constant', value=Rational(2, 1)))
            
            return AlgebraicExpression('fraction', numerator=numerator, denominator=denominator)
        
        return AlgebraicExpression('constant', value=Rational(0, 1))  # Default case
    
    @staticmethod
    def integrate(expr: AlgebraicExpression, var_name: str = 'x') -> AlgebraicExpression:
        """Compute symbolic indefinite integral (limited set)"""
        if expr.expr_type == 'constant':
            # integral c dx = c*x
            c = expr.value
            return AlgebraicExpression('product', 
                                     factors=[AlgebraicExpression('constant', value=c),
                                             AlgebraicExpression('variable', name=var_name)])
        
        elif expr.expr_type == 'variable':
            # integral x dx = x^2/2
            if expr.name == var_name:
                return AlgebraicExpression('fraction',
                                         numerator=AlgebraicExpression('power', 
                                                                     base=AlgebraicExpression('variable', name=var_name),
                                                                     exponent=AlgebraicExpression('constant', value=Rational(2, 1))),
                                         denominator=AlgebraicExpression('constant', value=Rational(2, 1)))
            else:
                # integral c dx = c*x (constant with respect to x)
                return AlgebraicExpression('product', 
                                         factors=[expr, AlgebraicExpression('variable', name=var_name)])
        
        elif expr.expr_type == 'power':
            # integral x^n dx = x^(n+1)/(n+1) for n ≠ -1
            if (expr.base.expr_type == 'variable' and expr.base.name == var_name and
                expr.exponent.expr_type == 'constant'):
                
                n = expr.exponent.value
                if hasattr(n, 'numerator'):
                    n_val = n.numerator / n.denominator
                else:
                    n_val = float(n)
                
                if n_val != -1:  # Avoid 1/x case for now
                    new_exponent = AlgebraicExpression('constant', value=Rational(int(n_val + 1), 1))
                    numerator = AlgebraicExpression('power', 
                                                  base=AlgebraicExpression('variable', name=var_name),
                                                  exponent=new_exponent)
                    denominator = AlgebraicExpression('constant', value=Rational(int(n_val + 1), 1))
                    
                    return AlgebraicExpression('fraction', numerator=numerator, denominator=denominator)
            
            # integral a^x dx = a^x/ln(a) - this is complex, return placeholder
            return AlgebraicExpression('constant', value=Rational(0, 1))  # Placeholder
        
        elif expr.expr_type == 'sum':
            # integral (f + g) dx = integral f dx + integral g dx (linearity)
            integrated_terms = [CalculusEngine.integrate(term, var_name) for term in expr.terms]
            return AlgebraicExpression('sum', terms=integrated_terms)
        
        elif expr.expr_type == 'fraction':
            # Handle special cases like integral 1/x dx = ln|x|
            if (expr.numerator.expr_type == 'constant' and 
                expr.denominator.expr_type == 'variable' and 
                expr.denominator.name == var_name and
                expr.numerator.value.equals(Rational(1, 1))):
                # integral 1/x dx = ln|x|
                return AlgebraicExpression('variable', name='ln')  # Simplified representation
            
            # integral c/f(x) dx - complex, return placeholder
            return AlgebraicExpression('constant', value=Rational(0, 1))  # Placeholder
        
        # For unsupported expressions, return placeholder
        return AlgebraicExpression('constant', value=Rational(0, 1))  # Placeholder with +C implied
    
    @staticmethod
    def numerical_integrate(func_expr: str, a: float, b: float, n: int = 100) -> FloatWithError:
        """Numerical integration using trapezoidal rule as fallback"""
        # This is a simplified implementation that evaluates string expressions
        # In a real implementation, you'd need a proper expression evaluator
        
        def f(x):
            # Very basic function evaluator for simple expressions
            if func_expr == 'x':
                return x
            elif func_expr == 'x^2':
                return x**2
            elif func_expr == 'sin(x)':
                import math
                return math.sin(x)
            elif func_expr == 'cos(x)':
                import math
                return math.cos(x)
            elif func_expr == 'e^x':
                import math
                return math.exp(x)
            else:
                return 0  # Placeholder
        
        # Trapezoidal rule
        h = (b - a) / n
        result = 0.5 * (f(a) + f(b))
        
        for i in range(1, n):
            result += f(a + i * h)
        
        result *= h
        
        # Estimate error (simplified)
        error_estimate = abs(b - a) * h**2 / 12  # Trapezoidal rule error bound
        
        return FloatWithError(result, error_estimate)


# ===== END HARD-CODED CALCULUS ENGINE =====

# ===== HARD-CODED LINEAR ALGEBRA ENGINE =====

class Vector:
    """Hard-coded vector implementation"""
    
    def __init__(self, components: List[float]):
        self.components = components
        self.dimension = len(components)
    
    def add(self, other: 'Vector') -> 'Vector':
        """Vector addition"""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        
        result_components = []
        for i in range(self.dimension):
            result_components.append(self.components[i] + other.components[i])
        
        return Vector(result_components)
    
    def scalar_multiply(self, scalar: float) -> 'Vector':
        """Scalar multiplication"""
        result_components = []
        for component in self.components:
            result_components.append(scalar * component)
        
        return Vector(result_components)
    
    def dot_product(self, other: 'Vector') -> float:
        """Dot product (inner product)"""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        
        result = 0.0
        for i in range(self.dimension):
            result += self.components[i] * other.components[i]
        
        return result
    
    def magnitude(self) -> float:
        """Vector magnitude (Euclidean norm)"""
        return (self.dot_product(self)) ** 0.5
    
    def normalize(self) -> 'Vector':
        """Return unit vector"""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        
        return self.scalar_multiply(1.0 / mag)
    
    def __str__(self):
        return f"[{', '.join(str(c) for c in self.components)}]"
    
    def __eq__(self, other: 'Vector') -> bool:
        if self.dimension != other.dimension:
            return False
        return all(abs(a - b) < 1e-10 for a, b in zip(self.components, other.components))


class Matrix:
    """Hard-coded matrix implementation"""
    
    def __init__(self, rows: List[List[float]]):
        self.rows = rows
        self.num_rows = len(rows)
        self.num_cols = len(rows[0]) if rows else 0
        
        # Validate rectangular matrix
        for row in rows:
            if len(row) != self.num_cols:
                raise ValueError("All rows must have the same number of columns")
    
    @staticmethod
    def identity(size: int) -> 'Matrix':
        """Create identity matrix"""
        rows = []
        for i in range(size):
            row = [1.0 if j == i else 0.0 for j in range(size)]
            rows.append(row)
        return Matrix(rows)
    
    @staticmethod
    def zero(rows: int, cols: int) -> 'Matrix':
        """Create zero matrix"""
        matrix_rows = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return Matrix(matrix_rows)
    
    def add(self, other: 'Matrix') -> 'Matrix':
        """Matrix addition"""
        if self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            raise ValueError("Matrix dimensions must match")
        
        result_rows = []
        for i in range(self.num_rows):
            result_row = []
            for j in range(self.num_cols):
                result_row.append(self.rows[i][j] + other.rows[i][j])
            result_rows.append(result_row)
        
        return Matrix(result_rows)
    
    def multiply(self, other: 'Matrix') -> 'Matrix':
        """Matrix multiplication"""
        if self.num_cols != other.num_rows:
            raise ValueError("Matrix dimensions incompatible for multiplication")
        
        result_rows = []
        for i in range(self.num_rows):
            result_row = []
            for j in range(other.num_cols):
                # Compute dot product of row i and column j
                sum_val = 0.0
                for k in range(self.num_cols):
                    sum_val += self.rows[i][k] * other.rows[k][j]
                result_row.append(sum_val)
            result_rows.append(result_row)
        
        return Matrix(result_rows)
    
    def scalar_multiply(self, scalar: float) -> 'Matrix':
        """Scalar multiplication"""
        result_rows = []
        for i in range(self.num_rows):
            result_row = []
            for j in range(self.num_cols):
                result_row.append(self.rows[i][j] * scalar)
            result_rows.append(result_row)
        
        return Matrix(result_rows)
    
    def transpose(self) -> 'Matrix':
        """Matrix transpose"""
        result_rows = []
        for j in range(self.num_cols):
            result_row = []
            for i in range(self.num_rows):
                result_row.append(self.rows[i][j])
            result_rows.append(result_row)
        
        return Matrix(result_rows)
    
    def determinant(self) -> float:
        """Matrix determinant (for 2x2 and 3x3 matrices)"""
        if self.num_rows != self.num_cols:
            raise ValueError("Determinant only defined for square matrices")
        
        if self.num_rows == 2:
            # 2x2 determinant: ad - bc
            a, b = self.rows[0]
            c, d = self.rows[1]
            return a * d - b * c
        
        elif self.num_rows == 3:
            # 3x3 determinant using rule of Sarrus or Laplace expansion
            a, b, c = self.rows[0]
            d, e, f = self.rows[1]
            g, h, i = self.rows[2]
            
            # Using Sarrus rule
            main_diagonal = a*e*i + b*f*g + c*d*h
            anti_diagonal = c*e*g + a*f*h + b*d*i
            
            return main_diagonal - anti_diagonal
        
        else:
            raise NotImplementedError("Determinant only implemented for 2x2 and 3x3 matrices")
    
    def inverse(self) -> 'Matrix':
        """Matrix inverse (for 2x2 and 3x3 matrices)"""
        if self.num_rows != self.num_cols:
            raise ValueError("Inverse only defined for square matrices")
        
        det = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular (not invertible)")
        
        if self.num_rows == 2:
            # 2x2 inverse
            a, b = self.rows[0]
            c, d = self.rows[1]
            
            # Inverse = (1/det) * [d, -b; -c, a]
            inv_rows = [
                [d / det, -b / det],
                [-c / det, a / det]
            ]
            return Matrix(inv_rows)
        
        elif self.num_rows == 3:
            # 3x3 inverse using adjugate matrix
            # This is complex, so for now return identity (placeholder)
            raise NotImplementedError("3x3 matrix inverse not yet implemented")
        
        else:
            raise NotImplementedError("Inverse only implemented for 2x2 matrices")
    
    def __str__(self):
        result = ""
        for row in self.rows:
            result += f"[{'  '.join(f'{x:.3f}' for x in row)}]\n"
        return result.strip()
    
    def __eq__(self, other: 'Matrix') -> bool:
        if self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            return False
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if abs(self.rows[i][j] - other.rows[i][j]) > 1e-10:
                    return False
        return True


class Tensor:
    """Symbolic tensor with index notation for contraction rules"""
    
    def __init__(self, name: str, indices: List[str]):
        self.name = name
        self.indices = indices  # List of index names (e.g., ['i', 'j', 'k'])
        self.rank = len(indices)
    
    def contract(self, other: 'Tensor', contraction_indices: List[tuple]) -> 'Tensor':
        """Tensor contraction following Einstein summation convention"""
        # contraction_indices is list of (self_index, other_index) pairs to contract
        
        remaining_indices = []
        for idx in self.indices:
            if not any(idx == pair[0] for pair in contraction_indices):
                remaining_indices.append(idx)
        
        for idx in other.indices:
            if not any(idx == pair[1] for pair in contraction_indices):
                remaining_indices.append(idx)
        
        # Create new tensor name representing the contraction
        new_name = f"({self.name}_{other.name})"
        
        return Tensor(new_name, remaining_indices)
    
    def __str__(self):
        indices_str = ''.join(self.indices)
        return f"{self.name}_{{{indices_str}}}"


class LinearAlgebraEngine:
    """Hard-coded linear algebra operations"""
    
    @staticmethod
    def vector_add(v1: Vector, v2: Vector) -> Vector:
        """Vector addition"""
        return v1.add(v2)
    
    @staticmethod
    def vector_scalar_multiply(v: Vector, scalar: float) -> Vector:
        """Vector scalar multiplication"""
        return v.scalar_multiply(scalar)
    
    @staticmethod
    def vector_dot_product(v1: Vector, v2: Vector) -> float:
        """Vector dot product"""
        return v1.dot_product(v2)
    
    @staticmethod
    def matrix_add(m1: Matrix, m2: Matrix) -> Matrix:
        """Matrix addition"""
        return m1.add(m2)
    
    @staticmethod
    def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
        """Matrix multiplication"""
        return m1.multiply(m2)
    
    @staticmethod
    def matrix_transpose(m: Matrix) -> Matrix:
        """Matrix transpose"""
        return m.transpose()
    
    @staticmethod
    def matrix_determinant(m: Matrix) -> float:
        """Matrix determinant"""
        return m.determinant()
    
    @staticmethod
    def matrix_vector_multiply(m: Matrix, v: Vector) -> Vector:
        """Matrix-vector multiplication"""
        if m.num_cols != v.dimension:
            raise ValueError("Matrix columns must match vector dimension")
        
        result_components = []
        for i in range(m.num_rows):
            sum_val = 0.0
            for j in range(m.num_cols):
                sum_val += m.rows[i][j] * v.components[j]
            result_components.append(sum_val)
        
        return Vector(result_components)
    
    @staticmethod
    def tensor_contract(t1: Tensor, t2: Tensor, contractions: List[tuple]) -> Tensor:
        """Tensor contraction"""
        return t1.contract(t2, contractions)


class DimensionalAnalysis:
    """Hard-coded dimensional analysis for physical consistency checking"""
    
    # Fundamental dimensions (base SI units)
    DIMENSIONS = {
        'L': 'length',           # meter (m)
        'M': 'mass',             # kilogram (kg)  
        'T': 'time',             # second (s)
        'Q': 'electric_charge',  # coulomb (C)
        'Θ': 'temperature',      # kelvin (K)
        'N': 'amount_substance', # mole (mol)
        'J': 'luminous_intensity' # candela (cd)
    }
    
    # Common derived units and their dimensional formulas
    DERIVED_UNITS = {
        # Mechanical
        'velocity': {'L': 1, 'T': -1},           # m/s
        'acceleration': {'L': 1, 'T': -2},       # m/s²
        'force': {'M': 1, 'L': 1, 'T': -2},      # N = kg·m/s²
        'energy': {'M': 1, 'L': 2, 'T': -2},     # J = kg·m²/s²
        'power': {'M': 1, 'L': 2, 'T': -3},      # W = kg·m²/s³
        'pressure': {'M': 1, 'L': -1, 'T': -2},  # Pa = kg/(m·s²)
        
        # Electrical
        'electric_field': {'M': 1, 'L': 1, 'T': -3, 'Q': -1},  # V/m = kg·m/(s³·C)
        'voltage': {'M': 1, 'L': 2, 'T': -3, 'Q': -1},          # V = kg·m²/(s³·C)
        'current': {'Q': 1, 'T': -1},                           # A = C/s
        'resistance': {'M': 1, 'L': 2, 'T': -3, 'Q': -2},      # Ω = kg·m²/(s³·C²)
        
        # Thermal
        'heat_capacity': {'M': 1, 'L': 2, 'T': -2, 'Θ': -1},   # J/K = kg·m²/(s²·K)
        
        # Dimensionless
        'dimensionless': {},                                    # pure number
        'angle': {},                                            # radians (dimensionless)
    }
    
    @staticmethod
    def create_dimension(dimensions: Dict[str, int]) -> Dict[str, int]:
        """Create a dimension dictionary from fundamental dimensions"""
        return dimensions.copy()
    
    @staticmethod
    def multiply_dimensions(dim1: Dict[str, int], dim2: Dict[str, int]) -> Dict[str, int]:
        """Multiply two dimensions (for multiplication of quantities)"""
        result = dim1.copy()
        for dim, power in dim2.items():
            result[dim] = result.get(dim, 0) + power
        return result
    
    @staticmethod
    def divide_dimensions(dim1: Dict[str, int], dim2: Dict[str, int]) -> Dict[str, int]:
        """Divide dimensions (for division of quantities)"""
        result = dim1.copy()
        for dim, power in dim2.items():
            result[dim] = result.get(dim, 0) - power
        return result
    
    @staticmethod
    def power_dimensions(dim: Dict[str, int], exponent: int) -> Dict[str, int]:
        """Raise dimensions to a power"""
        result = {}
        for d, power in dim.items():
            result[d] = power * exponent
        return result
    
    @staticmethod
    def root_dimensions(dim: Dict[str, int], root: int) -> Dict[str, int]:
        """Take nth root of dimensions (only if all powers divisible by root)"""
        result = {}
        for d, power in dim.items():
            if power % root != 0:
                raise ValueError(f"Cannot take {root}th root of dimension {d}^{power}")
            result[d] = power // root
        return result
    
    @staticmethod
    def dimensions_equal(dim1: Dict[str, int], dim2: Dict[str, int]) -> bool:
        """Check if two dimensions are equal"""
        # Remove zero powers
        d1 = {k: v for k, v in dim1.items() if v != 0}
        d2 = {k: v for k, v in dim2.items() if v != 0}
        return d1 == d2
    
    @staticmethod
    def format_dimensions(dimensions: Dict[str, int]) -> str:
        """Format dimensions as a readable string"""
        if not dimensions:
            return "dimensionless"
        
        parts = []
        for dim, power in sorted(dimensions.items()):
            if power == 0:
                continue
            elif power == 1:
                parts.append(dim)
            else:
                parts.append(f"{dim}^{power}")
        
        return "·".join(parts) if parts else "dimensionless"
    
    @staticmethod
    def check_equation_consistency(left_dims: List[Dict[str, int]], right_dims: List[Dict[str, int]]) -> Dict[str, Any]:
        """Check if an equation is dimensionally consistent"""
        result = {
            'consistent': True,
            'issues': [],
            'left_combined': None,
            'right_combined': None
        }
        
        # Combine dimensions on each side
        left_combined = left_dims[0] if left_dims else {}
        for dim in left_dims[1:]:
            left_combined = DimensionalAnalysis.multiply_dimensions(left_combined, dim)
        
        right_combined = right_dims[0] if right_dims else {}
        for dim in right_dims[1:]:
            right_combined = DimensionalAnalysis.multiply_dimensions(right_combined, dim)
        
        result['left_combined'] = left_combined
        result['right_combined'] = right_combined
        
        # Check consistency
        if not DimensionalAnalysis.dimensions_equal(left_combined, right_combined):
            result['consistent'] = False
            result['issues'].append({
                'type': 'dimension_mismatch',
                'message': f"Left side has dimensions {DimensionalAnalysis.format_dimensions(left_combined)}, right side has {DimensionalAnalysis.format_dimensions(right_combined)}",
                'severity': 'error'
            })
        
        return result
    
    @staticmethod
    def validate_physical_meaningfulness(dimensions: Dict[str, int]) -> Dict[str, Any]:
        """Check if dimensions make physical sense for common operations"""
        result = {
            'meaningful': True,
            'warnings': [],
            'issues': []
        }
        
        # Check for obviously nonsensical operations
        if dimensions.get('L', 0) < 0 and dimensions.get('T', 0) > 0:
            # Negative length with positive time - might indicate wrong operation
            result['warnings'].append("Negative length dimension with positive time may indicate incorrect operation")
        
        # Check for fractional dimensions that don't make sense
        fractional_dims = []
        for dim, power in dimensions.items():
            if power % 1 != 0:  # Non-integer power
                fractional_dims.append(f"{dim}^{power}")
        
        if fractional_dims:
            result['issues'].append({
                'type': 'fractional_dimension',
                'message': f"Fractional dimensions detected: {', '.join(fractional_dims)}. This may indicate invalid operations.",
                'severity': 'warning'
            })
        
        return result


# ===== END HARD-CODED DIMENSIONAL ANALYSIS =====

class MathCorrectionEngine:
    """Engine for detecting and correcting mathematical errors in code and app logic"""

    def __init__(self, memory_system):
        self.memory = memory_system
        self.arithmetic = IntegerArithmetic()
        self.algebraic_manipulator = AlgebraicManipulator()
        self.calculus = CalculusEngine()
        self.linear_algebra = LinearAlgebraEngine()

    def analyze_code_for_math_errors(self, code: str) -> Dict[str, Any]:
        """Analyze code for mathematical errors and suggest corrections"""
        analysis = {
            'errors_found': [],
            'corrections_suggested': [],
            'confidence': 0.0,
            'severity': 'low'
        }

        # Extract mathematical expressions from code
        math_expressions = self._extract_math_expressions(code)

        for expr_info in math_expressions:
            expr = expr_info['expression']
            line = expr_info['line']
            context = expr_info['context']

            # Analyze the expression
            error_analysis = self._analyze_expression(expr, context)

            if error_analysis['has_error']:
                analysis['errors_found'].append({
                    'expression': expr,
                    'line': line,
                    'error_type': error_analysis['error_type'],
                    'description': error_analysis['description'],
                    'severity': error_analysis['severity']
                })

                analysis['corrections_suggested'].append({
                    'original': expr,
                    'corrected': error_analysis['correction'],
                    'explanation': error_analysis['explanation'],
                    'line': line
                })

                # Update overall severity
                if error_analysis['severity'] == 'high':
                    analysis['severity'] = 'high'
                elif error_analysis['severity'] == 'medium' and analysis['severity'] != 'high':
                    analysis['severity'] = 'medium'

        analysis['confidence'] = len(analysis['errors_found']) / max(1, len(math_expressions))

        return analysis

    def _extract_math_expressions(self, code: str) -> List[Dict[str, Any]]:
        """Extract mathematical expressions from code"""
        expressions = []

        # Split code into lines
        lines = code.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Look for common mathematical patterns
            patterns = [
                r'(\w+)\s*=\s*([^;]+)',  # assignments
                r'return\s+([^;]+)',     # return statements
                r'if\s*\(([^)]+)\)',     # conditions
                r'(\w+)\s*([+\-*/])\s*=\s*([^;]+)',  # compound assignments
            ]

            for pattern in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        expr = ' '.join(match)
                    else:
                        expr = match

                    # Check if expression contains mathematical operations
                    if any(op in expr for op in ['+', '-', '*', '/', '=', '>', '<', '>=', '<=', '==']):
                        expressions.append({
                            'expression': expr.strip(),
                            'line': line_num,
                            'context': line
                        })

        return expressions

    def _analyze_expression(self, expr: str, context: str) -> Dict[str, Any]:
        """Analyze a single mathematical expression for errors"""
        result = {
            'has_error': False,
            'error_type': None,
            'description': '',
            'severity': 'low',
            'correction': expr,
            'explanation': ''
        }

        # Remove variable assignments for analysis
        clean_expr = re.sub(r'\w+\s*=\s*', '', expr)

        # Check for common mathematical errors

        # 1. Division by zero patterns
        if '/ 0' in clean_expr or '/0' in clean_expr:
            result['has_error'] = True
            result['error_type'] = 'division_by_zero'
            result['description'] = 'Division by zero detected'
            result['severity'] = 'high'
            result['correction'] = clean_expr.replace('/ 0', '/ 1').replace('/0', '/ 1')
            result['explanation'] = 'Replaced division by zero with division by one to prevent runtime error'

        # 2. Incorrect operator precedence (e.g., 2+3*4 vs (2+3)*4)
        if '+' in clean_expr and '*' in clean_expr and '(' not in clean_expr:
            result['has_error'] = True
            result['error_type'] = 'operator_precedence'
            result['description'] = 'Potential operator precedence issue'
            result['severity'] = 'medium'
            result['correction'] = f"({clean_expr})"
            result['explanation'] = 'Added parentheses to clarify operator precedence'

        # 3. Type mismatch in arithmetic
        if self._detects_type_mismatch(clean_expr):
            result['has_error'] = True
            result['error_type'] = 'type_mismatch'
            result['description'] = 'Potential type mismatch in arithmetic operation'
            result['severity'] = 'medium'
            result['correction'] = f"float({clean_expr})"
            result['explanation'] = 'Wrapped expression in float() to ensure numeric type consistency'

        # 4. Logical vs arithmetic operators
        if '==' in clean_expr and any(op in clean_expr for op in ['+', '-', '*', '/']):
            # Check if this might be a logical comparison where arithmetic was intended
            if re.search(r'\w+\s*==\s*\w+\s*[+\-*/]\s*\w+', clean_expr):
                result['has_error'] = True
                result['error_type'] = 'logical_arithmetic_confusion'
                result['description'] = 'Possible confusion between logical equality and arithmetic operations'
                result['severity'] = 'high'
                result['correction'] = clean_expr.replace('==', '=')
                result['explanation'] = 'Changed == to = for arithmetic assignment'

        return result

    def _detects_type_mismatch(self, expr: str) -> bool:
        """Detect potential type mismatches in expressions"""
        # Simple heuristic: if expression contains both numbers and strings
        has_numbers = bool(re.search(r'\d', expr))
        has_strings = bool(re.search(r'["\']', expr))

        return has_numbers and has_strings

    def generate_correction_capsule(self, error_info: Dict[str, Any]) -> 'Capsule':
        """Generate a capsule documenting a mathematical correction"""
        content = f"Mathematical Error Correction: {error_info['error_type']}\n"
        content += f"Original: {error_info['original']}\n"
        content += f"Corrected: {error_info['corrected']}\n"
        content += f"Explanation: {error_info['explanation']}\n"
        content += f"Line: {error_info['line']}"

        return Capsule(
            content=content,
            kind=CapsuleKind.MATH_ERROR,
            certainty=0.9,
            relevance="code_correction",
            perspective="mathematical_analysis"
        )


class CognitiveLayer:
    """High-level cognitive processing layer - makes Cayde feel like a thinking companion"""

    def __init__(self, memory_system):
        self.memory = memory_system
        self.pedagogical_engine = PedagogicalEngine()
        self.theory_evaluator = TheoryEvaluator()
        self.epistemological_framework = EpistemologicalFramework()
        self.communication_stylist = CommunicationStylist()
        self.math_correction_engine = MathCorrectionEngine(memory_system)

    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query through the full cognitive stack"""
        # Gather relevant capsules and context
        relevant_capsules = self._gather_relevant_context(query)

        # Apply cognitive processing
        pedagogical_analysis = self.pedagogical_engine.analyze_query_complexity(query, relevant_capsules)
        theory_assessment = self.theory_evaluator.assess_relevant_theories(relevant_capsules)
        uncertainty_analysis = self.epistemological_framework.assess_knowledge_limits(query, relevant_capsules)

        # Check for mathematical errors in code/app development context
        math_correction_analysis = self._analyze_for_math_errors(query, context)

        # Generate response through communication stylist
        response = self.communication_stylist.generate_response(
            query=query,
            pedagogical_analysis=pedagogical_analysis,
            theory_assessment=theory_assessment,
            uncertainty_analysis=uncertainty_analysis,
            context=context or {}
        )

        # If math errors were found, enhance the response with corrections
        if math_correction_analysis['errors_found']:
            response = self._enhance_response_with_math_corrections(response, math_correction_analysis)

        return {
            'response': response,
            'cognitive_analysis': {
                'pedagogical': pedagogical_analysis,
                'theoretical': theory_assessment,
                'epistemological': uncertainty_analysis,
                'math_correction': math_correction_analysis
            },
            'relevant_capsules': relevant_capsules
        }

    def _gather_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """Gather relevant capsules and context for the query"""
        # Simple relevance gathering - in practice this would be more sophisticated
        relevant = []

        for capsule in self.memory.capsules:
            # Check for keyword matches and semantic relevance
            query_lower = query.lower()
            content_lower = capsule.content.lower()

            relevance_score = 0

            # Keyword matching
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                relevance_score += overlap * 0.5

            # Concept matching (simplified)
            key_concepts = ['physics', 'mathematics', 'theory', 'law', 'principle', 'equation']
            for concept in key_concepts:
                if concept in query_lower and concept in content_lower:
                    relevance_score += 1.0

            if relevance_score > 0.5:  # Relevance threshold
                relevant.append({
                    'capsule': capsule,
                    'relevance_score': relevance_score,
                    'content_preview': capsule.content[:100] + "..." if len(capsule.content) > 100 else capsule.content
                })

        # Sort by relevance
        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant[:5]  # Top 5 most relevant

    def _analyze_for_math_errors(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze query for potential mathematical errors in code/app development context"""
        # Check if this looks like code or app development discussion
        code_indicators = ['function', 'def ', 'class ', 'if ', 'for ', 'while ', 'return ', 'import ', 'from ', 'print(',
                          'variable', 'calculate', 'compute', 'algorithm', 'app', 'application', 'code', 'program']

        query_lower = query.lower()
        is_code_context = any(indicator in query_lower for indicator in code_indicators)

        # Also check for mathematical operators in the query
        has_math_ops = any(op in query for op in ['+', '-', '*', '/', '=', '>', '<', '>=', '<=', '=='])

        if not (is_code_context or has_math_ops):
            return {'errors_found': [], 'corrections_suggested': [], 'confidence': 0.0, 'severity': 'none'}

        # Extract code from query if present
        code_snippet = self._extract_code_from_query(query)

        if code_snippet:
            return self.math_correction_engine.analyze_code_for_math_errors(code_snippet)
        else:
            # Try to analyze mathematical expressions in the query itself
            return self.math_correction_engine.analyze_code_for_math_errors(query)

    def _extract_code_from_query(self, query: str) -> str:
        """Extract code snippets from a natural language query"""
        # Look for code blocks marked with ``` or indented code
        code_blocks = re.findall(r'```(?:python)?\n?(.*?)\n?```', query, re.DOTALL)
        if code_blocks:
            return '\n'.join(code_blocks)

        # Look for indented code blocks
        lines = query.split('\n')
        code_lines = []
        in_code_block = False

        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line.strip())
                in_code_block = True
            elif in_code_block and line.strip():
                # Continue if we're in a code block
                code_lines.append(line)
            elif in_code_block and not line.strip():
                # Empty line ends code block
                break

        if code_lines:
            return '\n'.join(code_lines)

        return ""

    def _enhance_response_with_math_corrections(self, base_response: str, math_analysis: Dict[str, Any]) -> str:
        """Enhance the response with mathematical corrections"""
        if not math_analysis['errors_found']:
            return base_response

        correction_text = "\n\nMathematical Corrections Detected:\n"

        for i, error in enumerate(math_analysis['errors_found'], 1):
            correction_text += f"\n{i}. **{error['error_type'].replace('_', ' ').title()}** (Line {error['line']})\n"
            correction_text += f"   Problem: {error['description']}\n"
            correction_text += f"   Severity: {error['severity']}\n"

        correction_text += "\nSuggested Corrections:\n"

        for i, correction in enumerate(math_analysis['corrections_suggested'], 1):
            correction_text += f"\n{i}. **Before:** `{correction['original']}`\n"
            correction_text += f"   **After:** `{correction['corrected']}`\n"
            correction_text += f"   **Why:** {correction['explanation']}\n"

            # Create and store correction capsule
            correction_capsule = self.math_correction_engine.generate_correction_capsule({
                'original': correction['original'],
                'corrected': correction['corrected'],
                'explanation': correction['explanation'],
                'line': correction['line'],
                'error_type': math_analysis['errors_found'][i-1]['error_type']
            })
            self.memory.add_capsule(correction_capsule)

        correction_text += f"\nAnalysis Summary: {len(math_analysis['errors_found'])} errors found, severity: {math_analysis['severity']}"

        return base_response + correction_text


class PedagogicalEngine:
    """Engine for explaining hard ideas simply without dumbing them down"""

    def analyze_query_complexity(self, query: str, relevant_capsules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the complexity of a query and how to explain it"""
        analysis = {
            'complexity_level': self._assess_complexity(query),
            'core_concepts': self._extract_core_concepts(query, relevant_capsules),
            'explanation_strategy': 'progressive_reveal',
            'analogies_available': [],
            'prerequisites': []
        }

        # Assess what background knowledge is needed
        analysis['prerequisites'] = self._identify_prerequisites(analysis['core_concepts'])

        # Find good analogies
        analysis['analogies_available'] = self._find_analogies(analysis['core_concepts'])

        return analysis

    def _assess_complexity(self, query: str) -> str:
        """Assess the complexity level of a query"""
        complexity_indicators = {
            'high': ['quantum', 'relativity', 'field theory', 'topology', 'abstract algebra'],
            'medium': ['calculus', 'thermodynamics', 'electromagnetism', 'mechanics'],
            'low': ['basic physics', 'arithmetic', 'geometry basics']
        }

        query_lower = query.lower()
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return level

        return 'medium'  # Default

    def _extract_core_concepts(self, query: str, relevant_capsules: List[Dict[str, Any]]) -> List[str]:
        """Extract the core concepts that need to be explained"""
        concepts = []

        # From query
        concept_keywords = [
            'energy', 'momentum', 'force', 'mass', 'velocity', 'acceleration',
            'quantum', 'relativity', 'gravity', 'electromagnetism', 'thermodynamics',
            'entropy', 'probability', 'information', 'consciousness'
        ]

        query_lower = query.lower()
        for concept in concept_keywords:
            if concept in query_lower:
                concepts.append(concept)

        # From relevant capsules
        for item in relevant_capsules:
            capsule_content = item['capsule'].content.lower()
            for concept in concept_keywords:
                if concept in capsule_content and concept not in concepts:
                    concepts.append(concept)

        return concepts

    def _identify_prerequisites(self, core_concepts: List[str]) -> List[str]:
        """Identify prerequisite knowledge needed"""
        prerequisites_map = {
            'energy': ['basic physics', 'conservation laws'],
            'quantum': ['wave-particle duality', 'uncertainty principle', 'probability'],
            'relativity': ['special relativity', 'time dilation', 'length contraction'],
            'thermodynamics': ['heat', 'work', 'temperature', 'entropy'],
            'electromagnetism': ['electric fields', 'magnetic fields', 'Maxwell equations']
        }

        prerequisites = []
        for concept in core_concepts:
            if concept in prerequisites_map:
                prerequisites.extend(prerequisites_map[concept])

        return list(set(prerequisites))  # Remove duplicates

    def _find_analogies(self, core_concepts: List[str]) -> List[Dict[str, str]]:
        """Find good analogies for explaining concepts"""
        analogies = []

        analogy_map = {
            'energy': {
                'analogy': 'money in a bank account',
                'explanation': 'Energy is like money - it can change forms but total amount is conserved'
            },
            'quantum': {
                'analogy': 'pixels on a computer screen',
                'explanation': 'Just as a digital image is made of discrete pixels, quantum mechanics shows reality is made of discrete quanta'
            },
            'relativity': {
                'analogy': 'different reference frames in a train',
                'explanation': 'Time and space measurements depend on your relative motion, like how time seems to pass differently when you\'re moving'
            },
            'entropy': {
                'analogy': 'messy room getting messier',
                'explanation': 'Entropy is like the natural tendency of things to become more disordered, like how a tidy room inevitably becomes messy'
            }
        }

        for concept in core_concepts:
            if concept in analogy_map:
                analogies.append({
                    'concept': concept,
                    'analogy': analogy_map[concept]['analogy'],
                    'explanation': analogy_map[concept]['explanation']
                })

        return analogies


class TheoryEvaluator:
    """Engine for judging theories by impact, not fashion"""

    def assess_relevant_theories(self, relevant_capsules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess theories based on their real impact rather than popularity"""
        assessment = {
            'theories_evaluated': [],
            'impact_assessment': {},
            'paradigm_shifting_potential': {},
            'resistance_to_fashion': {}
        }

        for item in relevant_capsules:
            capsule = item['capsule']
            if capsule.kind.name == 'THEORY':
                theory_assessment = self._assess_theory_impact(capsule)
                assessment['theories_evaluated'].append(capsule.content[:50] + "...")
                assessment['impact_assessment'][capsule.content[:30]] = theory_assessment

        return assessment

    def _assess_theory_impact(self, capsule) -> Dict[str, Any]:
        """Assess the real impact of a theory"""
        impact = {
            'historical_significance': self._calculate_historical_impact(capsule),
            'predictive_power': self._assess_predictive_power(capsule),
            'unification_potential': self._measure_unification(capsule),
            'experimental_validation': self._check_experimental_support(capsule),
            'paradigm_shifting': self._assess_paradigm_shift(capsule),
            'fashion_resistance': self._measure_fashion_resistance(capsule)
        }

        # Overall impact score
        impact['overall_impact'] = (
            impact['historical_significance'] * 0.25 +
            impact['predictive_power'] * 0.25 +
            impact['unification_potential'] * 0.2 +
            impact['experimental_validation'] * 0.15 +
            impact['paradigm_shifting'] * 0.1 +
            impact['fashion_resistance'] * 0.05
        )

        return impact

    def _calculate_historical_impact(self, capsule) -> float:
        """Calculate historical significance (0-1 scale)"""
        # Simplified assessment based on content indicators
        content = capsule.content.lower()

        high_impact_indicators = ['einstein', 'newton', 'quantum', 'relativity', 'evolution']
        medium_impact_indicators = ['thermodynamics', 'electromagnetism', 'gravity']

        if any(indicator in content for indicator in high_impact_indicators):
            return 0.9
        elif any(indicator in content for indicator in medium_impact_indicators):
            return 0.7
        else:
            return 0.5

    def _assess_predictive_power(self, capsule) -> float:
        """Assess how well the theory predicts new phenomena"""
        content = capsule.content.lower()

        # Theories with strong predictive power
        predictive_theories = ['relativity', 'quantum mechanics', 'evolution']
        if any(theory in content for theory in predictive_theories):
            return 0.9
        else:
            return 0.6

    def _measure_unification(self, capsule) -> float:
        """Measure how well the theory unifies different phenomena"""
        content = capsule.content.lower()

        # Unifying theories
        unifying_concepts = ['unified field', 'grand unification', 'electroweak', 'standard model']
        if any(concept in content for concept in unifying_concepts):
            return 0.9
        elif 'unified' in content or 'unifies' in content:
            return 0.8
        else:
            return 0.4

    def _check_experimental_support(self, capsule) -> float:
        """Check experimental validation strength"""
        # Simplified - in practice would check against experimental database
        if capsule.success_status == "proven":
            return 0.9
        elif capsule.certainty > 0.8:
            return 0.8
        elif capsule.certainty > 0.6:
            return 0.6
        else:
            return 0.4

    def _assess_paradigm_shift(self, capsule) -> float:
        """Assess paradigm-shifting potential"""
        content = capsule.content.lower()

        paradigm_shifters = ['relativity', 'quantum', 'evolution', 'heliocentrism']
        if any(shifter in content for shifter in paradigm_shifters):
            return 0.95
        else:
            return 0.5

    def _measure_fashion_resistance(self, capsule) -> float:
        """Measure resistance to intellectual fashion/trends"""
        # Theories that go against prevailing wisdom tend to have higher resistance
        content = capsule.content.lower()

        counter_intuitive = ['quantum', 'relativity', 'uncertainty', 'observer effect']
        if any(concept in content for concept in counter_intuitive):
            return 0.9  # High resistance to fashion
        else:
            return 0.6


class EpistemologicalFramework:
    """Framework for respecting mystery without hallucinating certainty"""

    def assess_knowledge_limits(self, query: str, relevant_capsules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the limits of our knowledge for this query"""
        assessment = {
            'known_knowns': [],
            'known_unknowns': [],
            'unknown_unknowns': [],
            'uncertainty_quantification': 0.0,
            'humility_recommendations': [],
            'mysteries_to_respect': []
        }

        # Analyze what we know vs don't know
        query_concepts = self._extract_query_concepts(query)

        for concept in query_concepts:
            knowledge_state = self._assess_knowledge_state(concept, relevant_capsules)

            if knowledge_state['confidence'] > 0.8:
                assessment['known_knowns'].append(concept)
            elif knowledge_state['confidence'] > 0.3:
                assessment['known_unknowns'].append({
                    'concept': concept,
                    'known_aspects': knowledge_state['known_aspects'],
                    'unknown_aspects': knowledge_state['unknown_aspects']
                })
            else:
                assessment['unknown_unknowns'].append(concept)

        # Identify mysteries that should be respected
        assessment['mysteries_to_respect'] = self._identify_respectable_mysteries(query_concepts)

        # Generate humility recommendations
        assessment['humility_recommendations'] = self._generate_humility_guidance(assessment)

        # Overall uncertainty quantification
        assessment['uncertainty_quantification'] = self._quantify_overall_uncertainty(assessment)

        return assessment

    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query"""
        concepts = []
        query_lower = query.lower()

        key_concepts = [
            'consciousness', 'free will', 'quantum gravity', 'dark matter', 'dark energy',
            'origin of life', 'meaning of life', 'nature of reality', 'time', 'infinity',
            'information', 'computation', 'intelligence'
        ]

        for concept in key_concepts:
            if concept in query_lower:
                concepts.append(concept)

        return concepts

    def _assess_knowledge_state(self, concept: str, relevant_capsules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess our knowledge state for a given concept"""
        knowledge_state = {
            'confidence': 0.5,
            'known_aspects': [],
            'unknown_aspects': []
        }

        # Check capsules for information about this concept
        relevant_info = []
        for item in relevant_capsules:
            content = item['capsule'].content.lower()
            if concept in content:
                relevant_info.append(item['capsule'])

        # Assess confidence based on available information
        if len(relevant_info) > 2:
            knowledge_state['confidence'] = 0.8
            knowledge_state['known_aspects'] = ['basic principles', 'some applications']
        elif len(relevant_info) > 0:
            knowledge_state['confidence'] = 0.6
            knowledge_state['known_aspects'] = ['partial understanding']
            knowledge_state['unknown_aspects'] = ['complete picture', 'underlying mechanisms']
        else:
            knowledge_state['confidence'] = 0.2
            knowledge_state['unknown_aspects'] = ['fundamental nature', 'basic principles']

        return knowledge_state

    def _identify_respectable_mysteries(self, concepts: List[str]) -> List[Dict[str, str]]:
        """Identify mysteries that should be respected rather than 'solved'"""
        mysteries = []

        mystery_map = {
            'consciousness': {
                'mystery': 'Why does consciousness exist?',
                'respect_reason': 'We experience it directly but cannot explain its emergence from physical processes'
            },
            'free_will': {
                'mystery': 'Do we have genuine free will?',
                'respect_reason': 'The question remains open despite centuries of philosophical debate'
            },
            'quantum_gravity': {
                'mystery': 'How do gravity and quantum mechanics reconcile?',
                'respect_reason': 'Current theories break down at Planck scales - we simply don\'t know'
            },
            'origin_of_life': {
                'mystery': 'How did life emerge from non-living matter?',
                'respect_reason': 'We have theories but no definitive answer - the transition remains mysterious'
            },
            'meaning_of_life': {
                'mystery': 'What is the meaning of life?',
                'respect_reason': 'This is a deeply personal question that transcends scientific reductionism'
            }
        }

        for concept in concepts:
            if concept in mystery_map:
                mysteries.append(mystery_map[concept])

        return mysteries

    def _generate_humility_guidance(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate guidance for maintaining intellectual humility"""
        guidance = []

        if assessment['unknown_unknowns']:
            guidance.append("Remember that we don't even know what we don't know about these topics")

        if len(assessment['known_unknowns']) > len(assessment['known_knowns']):
            guidance.append("We know more about what we don't know than what we do know")

        if assessment['mysteries_to_respect']:
            guidance.append("Some questions may be inherently mysterious - respect them rather than forcing answers")

        if not guidance:
            guidance.append("Maintain healthy skepticism even about well-established knowledge")

        return guidance

    def _quantify_overall_uncertainty(self, assessment: Dict[str, Any]) -> float:
        """Quantify overall uncertainty level"""
        total_concepts = len(assessment['known_knowns']) + len(assessment['known_unknowns']) + len(assessment['unknown_unknowns'])

        if total_concepts == 0:
            return 0.5

        # Weight uncertainty by knowledge state
        uncertainty_score = (
            len(assessment['known_knowns']) * 0.2 +      # Low uncertainty
            len(assessment['known_unknowns']) * 0.6 +    # Medium uncertainty
            len(assessment['unknown_unknowns']) * 0.9    # High uncertainty
        ) / total_concepts

        return uncertainty_score


class CommunicationStylist:
    """Engine for authentic communication that sounds like understanding, not recitation"""

    def generate_response(self, query: str, pedagogical_analysis: Dict[str, Any],
                         theory_assessment: Dict[str, Any], uncertainty_analysis: Dict[str, Any],
                         context: Dict[str, Any]) -> str:
        """Generate a response that sounds like a thinking companion"""

        # Build response components
        response_parts = []

        # Opening - acknowledge the query thoughtfully
        response_parts.append(self._generate_thoughtful_opening(query, pedagogical_analysis))

        # Main explanation - use pedagogical analysis
        if pedagogical_analysis['core_concepts']:
            response_parts.append(self._generate_concept_explanation(pedagogical_analysis))

        # Theory assessment - if relevant
        if theory_assessment['theories_evaluated']:
            response_parts.append(self._generate_theory_assessment(theory_assessment))

        # Uncertainty and humility - always include
        response_parts.append(self._generate_uncertainty_expression(uncertainty_analysis))

        # Closing - thoughtful reflection
        response_parts.append(self._generate_reflective_closing(query, uncertainty_analysis))

        # Combine and add personality
        full_response = " ".join(response_parts)
        return self._add_personality_flavor(full_response, context)

    def _generate_thoughtful_opening(self, query: str, pedagogical_analysis: Dict[str, Any]) -> str:
        """Generate a thoughtful opening that shows understanding"""
        complexity = pedagogical_analysis['complexity_level']

        openings = {
            'high': [
                "This touches on some of the deepest mysteries we've encountered...",
                "You're asking about concepts that have puzzled great minds for generations...",
                "This question gets to the heart of some truly profound ideas..."
            ],
            'medium': [
                "This is a fascinating question that reveals important connections...",
                "You've identified an interesting aspect of how things work...",
                "This touches on some elegant principles that govern our reality..."
            ],
            'low': [
                "This is a great question that helps clarify fundamental concepts...",
                "You've asked about something essential that builds our understanding...",
                "This question reveals the beautiful logic underlying natural phenomena..."
            ]
        }

        import random
        return random.choice(openings[complexity])

    def _generate_concept_explanation(self, pedagogical_analysis: Dict[str, Any]) -> str:
        """Generate explanation of core concepts"""
        concepts = pedagogical_analysis['core_concepts']
        analogies = pedagogical_analysis['analogies_available']

        explanation = f"When we talk about {', '.join(concepts)}, we're dealing with ideas that "

        if analogies:
            analogy = analogies[0]  # Use first available analogy
            explanation += f"are a bit like {analogy['analogy']}. {analogy['explanation']} "
        else:
            explanation += "connect fundamental aspects of how the universe operates. "

        # Add prerequisites if needed
        prereqs = pedagogical_analysis['prerequisites']
        if prereqs:
            explanation += f"To really appreciate this, it's helpful to understand {', '.join(prereqs[:2])}. "

        return explanation

    def _generate_theory_assessment(self, theory_assessment: Dict[str, Any]) -> str:
        """Generate assessment of relevant theories"""
        if not theory_assessment['impact_assessment']:
            return ""

        # Find the theory with highest impact
        best_theory = max(
            theory_assessment['impact_assessment'].items(),
            key=lambda x: x[1]['overall_impact']
        )

        theory_name = best_theory[0]
        impact_score = best_theory[1]['overall_impact']

        if impact_score > 0.8:
            assessment = f"The framework we're discussing here, {theory_name}, has proven remarkably powerful - not just as an intellectual achievement, but as a tool that actually works in our attempts to understand the world."
        elif impact_score > 0.6:
            assessment = f"This approach, {theory_name}, has shown real value in helping us make sense of complex phenomena, even if it has its limitations."
        else:
            assessment = f"While {theory_name} offers some insights, its ultimate significance remains to be seen as we continue exploring these questions."

        return assessment

    def _generate_uncertainty_expression(self, uncertainty_analysis: Dict[str, Any]) -> str:
        """Generate expression of appropriate uncertainty and humility"""
        uncertainty_level = uncertainty_analysis['uncertainty_quantification']
        mysteries = uncertainty_analysis['mysteries_to_respect']
        guidance = uncertainty_analysis['humility_recommendations']

        if uncertainty_level > 0.7:
            uncertainty_text = "That said, we have to acknowledge how much we still don't understand. "
        elif uncertainty_level > 0.4:
            uncertainty_text = "Of course, our understanding here is incomplete and evolving. "
        else:
            uncertainty_text = "While we have some solid ground to stand on here, "
        if mysteries:
            mystery = mysteries[0]
            uncertainty_text += f"questions like '{mystery['mystery']}' remind us that {mystery['respect_reason'].lower()}. "

        if guidance:
            uncertainty_text += f"{guidance[0]} "

        return uncertainty_text

    def _generate_reflective_closing(self, query: str, uncertainty_analysis: Dict[str, Any]) -> str:
        """Generate a reflective closing"""
        closings = [
            "It's conversations like this that keep pushing our understanding forward.",
            "These are the kinds of questions that make thinking about reality so worthwhile.",
            "Exploring these ideas together is what makes the journey meaningful.",
            "Questions like yours are what drive genuine intellectual progress.",
            "This kind of thoughtful inquiry is how we gradually illuminate the mysteries around us."
        ]

        import random
        return random.choice(closings)

    def _add_personality_flavor(self, response: str, context: Dict[str, Any]) -> str:
        """Add personality-driven flavor to make it sound authentic"""
        # Add Cayde's personality traits to the response
        # This would integrate with the personality system

        # For now, add some thoughtful pauses and authentic language
        authentic_phrases = [
            ("I think", "It seems to me"),
            ("we find", "we discover"),
            ("suggests", "hints at"),
            ("appears to", "seems to"),
            ("tends to", "often"),
        ]

        # Apply some personality flavor
        flavored_response = response

        # Add occasional reflective phrases
        if "think" in response and random.random() < 0.3:
            flavored_response = flavored_response.replace("I think", "You know, I think")

        return flavored_response


# ===== END COGNITIVE LAYER =====

class CapsuleKind(Enum):
    FACT = "fact"
    CONCEPT = "concept"
    EVENT = "event"
    PERSON = "person"
    THEORY = "theory"
    METHOD = "method"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    AUDIO = "audio"
    MATH_ERROR = "math_error"
    CODE_ANALYSIS = "code_analysis"

@dataclass
class Capsule:
    content: str
    embedding: Optional[np.ndarray] = None
    perspective: str = "user"
    certainty: float = 0.5
    relevance: str = "general"
    character: Optional[str] = None
    persona: Optional[str] = None
    kind: CapsuleKind = CapsuleKind.CONCEPT
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    pose: Dict[str, Any] = field(default_factory=lambda: {
        "temporal": None,
        "perspective": "user",
        "certainty": 0.5,
        "attention": 0.5,
        "relevance": 0.5,
        "abstraction": 0.5
    })
    gravity: float = 0.5
    orbit_radius: float = 1.0
    links: List['Capsule'] = field(default_factory=list)
    locked: bool = False
    parent: Optional['Capsule'] = None
    # Temporal sequencing
    temporal_order: int = field(default_factory=lambda: int(time.time() * 1000))
    # Enhanced temporal and trajectory tracking
    confidence_history: List[Tuple[int, float]] = field(default_factory=list)  # (temporal_order, confidence)
    paradigm_shifts: List[Dict[str, Any]] = field(default_factory=list)  # Major changes in understanding
    intellectual_conflicts: List[Dict[str, Any]] = field(default_factory=list)  # Tensions between ideas
    # Cayde personality tracking
    success_status: Optional[str] = None  # "success", "failure", "proven_later", "archived"
    proven_by: Optional[str] = None  # Scientist who later proved this theory
    insight_potential: float = 0.0  # For breakthrough detection

    def __post_init__(self):
        if self.embedding is None:
            try:
                if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                    self.embedding = cp.random.rand(128).get()
                elif GPU_TYPE == "AMD":
                    import torch
                    self.embedding = torch.rand(128).cuda().cpu().numpy()
            except Exception:
                self.embedding = np.random.rand(128)

    def update_pose(self, delta_pose: Dict[str, float]):
        """Update capsule pose parameters"""
        for k, v in delta_pose.items():
            if k in self.pose:
                self.pose[k] = (self.pose[k] + v) / 2

    def merge_with(self, other: 'Capsule'):
        """Merge with another capsule using creative synthesis"""
        if self.locked or other.locked:
            return

        # Creative merging: generate synthetic content from embeddings
        try:
            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                emb1_gpu = cp.asarray(self.embedding)
                emb2_gpu = cp.asarray(other.embedding)
                # Create novel combination by weighted interpolation with noise
                alpha = 0.5 + (np.random.rand() - 0.5) * 0.3  # Random weight between 0.35-0.65
                combined_emb = alpha * emb1_gpu + (1 - alpha) * emb2_gpu
                # Add creative noise
                noise = cp.random.normal(0, 0.1, combined_emb.shape)
                combined_emb += noise
                self.embedding = combined_emb.get()
            elif GPU_TYPE == "AMD":
                import torch
                emb1_gpu = torch.from_numpy(self.embedding).cuda()
                emb2_gpu = torch.from_numpy(other.embedding).cuda()
                alpha = 0.5 + (torch.rand(1).cuda() - 0.5) * 0.3
                combined_emb = alpha * emb1_gpu + (1 - alpha) * emb2_gpu
                noise = torch.normal(0, 0.1, combined_emb.shape).cuda()
                combined_emb += noise
                self.embedding = combined_emb.cpu().numpy()
            else:
                # CPU fallback with creative combination
                alpha = 0.5 + (np.random.rand() - 0.5) * 0.3
                combined_emb = alpha * self.embedding + (1 - alpha) * other.embedding
                noise = np.random.normal(0, 0.1, combined_emb.shape)
                combined_emb += noise
                self.embedding = combined_emb
        except Exception:
            # Fallback to simple averaging
            self.embedding = (self.embedding + other.embedding) / 2

        # Generate synthetic content based on capsule types and characters
        content_parts = []

        # Extract key concepts from both contents
        self_words = set(self.content.lower().split())
        other_words = set(other.content.lower().split())
        common_words = self_words.intersection(other_words)
        unique_self = self_words - common_words
        unique_other = other_words - common_words

        # Create synthetic description
        if self.character and other.character and self.character != other.character:
            content_parts.append(f"Synthesis of {self.character}'s {self.content.split()[0]} and {other.character}'s {other.content.split()[0]}")
        elif self.kind == CapsuleKind.THEORY and other.kind == CapsuleKind.OBSERVATION:
            content_parts.append(f"Theoretical framework combining {self.content[:30]}... with empirical evidence from {other.content[:30]}...")
        elif self.kind == CapsuleKind.METHOD and other.kind == CapsuleKind.CONCEPT:
            content_parts.append(f"Methodological approach applying {self.content[:25]}... to conceptual framework of {other.content[:25]}...")
        else:
            # Default creative synthesis
            if common_words:
                content_parts.append(f"Integrated understanding: {', '.join(list(common_words)[:3])}")
            if unique_self:
                content_parts.append(f"Enhanced by: {', '.join(list(unique_self)[:2])}")
            if unique_other:
                content_parts.append(f"Combined with: {', '.join(list(unique_other)[:2])}")

        self.content = " | ".join(content_parts) if content_parts else f"Creative synthesis: {self.content[:40]}... + {other.content[:40]}..."

        # Update temporal ordering (take the later one)
        self.temporal_order = max(self.temporal_order, other.temporal_order)

        # Merge success/failure status
        if self.success_status == "failure" and other.success_status == "success" and other.character != self.character:
            self.success_status = "proven_later"
            self.proven_by = other.character
            self.insight_potential = min(1.0, self.insight_potential + 0.3)
        elif other.success_status == "proven_later":
            self.success_status = "proven_later"
            self.proven_by = other.proven_by
            self.insight_potential = min(1.0, self.insight_potential + 0.2)

        # Update other properties
        self.gravity = max(self.gravity, other.gravity)
        self.orbit_radius = (self.orbit_radius + other.orbit_radius) / 2
        self.links += other.links
        self.insight_potential = (self.insight_potential + other.insight_potential) / 2

    def influence(self, other: 'Capsule', strength: float = 0.1):
        """Influence another capsule if locked"""
        if not self.locked:
            return
        other.gravity = min(1.0, other.gravity + strength)
        other.pose["attention"] = min(1.0, other.pose["attention"] + strength * 0.5)

    def split(self, new_content: str) -> 'Capsule':
        """Create a child capsule"""
        child = Capsule(
            content=new_content,
            embedding=self.embedding.copy(),
            perspective=self.pose["perspective"],
            character=self.character,
            persona=self.persona
        )
        child.parent = self
        child.gravity = self.gravity * 0.8
        child.orbit_radius = self.orbit_radius + 0.2
        return child

    def update_confidence(self, new_confidence: float):
        """Update confidence and track history for intellectual trajectory"""
        current_time = int(time.time() * 1000)
        self.confidence_history.append((current_time, new_confidence))
        self.pose["certainty"] = new_confidence

        # Detect paradigm shifts (sudden confidence changes)
        if len(self.confidence_history) >= 3:
            recent = self.confidence_history[-3:]
            change_rate = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
            if abs(change_rate) > 0.01:  # Significant confidence shift
                self.paradigm_shifts.append({
                    "timestamp": current_time,
                    "old_confidence": recent[0][1],
                    "new_confidence": recent[-1][1],
                    "change_rate": change_rate,
                    "trigger": "confidence_shift"
                })

    def detect_intellectual_conflict(self, other: 'Capsule') -> bool:
        """Detect conflicts between ideas that drive intellectual growth"""
        if not self.character or not other.character:
            return False

        # Same scientist, different time periods = evolution of thought
        if self.character == other.character and abs(self.temporal_order - other.temporal_order) > 3600000:  # 1 hour apart
            conflict_score = 1 - self._calculate_similarity(other)
            if conflict_score > 0.7:  # High conflict = paradigm shift potential
                self.intellectual_conflicts.append({
                    "type": "self_revision",
                    "other_capsule": other.uuid,
                    "conflict_score": conflict_score,
                    "temporal_gap": abs(self.temporal_order - other.temporal_order)
                })
                return True

        # Different scientists challenging established ideas
        if self.success_status == "success" and other.success_status == "failure" and self.character != other.character:
            self.intellectual_conflicts.append({
                "type": "paradigm_challenge",
                "challenger": other.character,
                "established": self.character,
                "conflict_score": self._calculate_similarity(other)
            })
            return True

        return False

    def _calculate_similarity(self, other_capsule: 'Capsule') -> float:
        """Calculate semantic similarity between capsules"""
        if not hasattr(other_capsule, 'content') or not self.content:
            return 0.0

        # Use embeddings if available
        if self.embedding is not None and other_capsule.embedding is not None:
            try:
                if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                    # GPU-accelerated cosine similarity
                    a_gpu = cp.asarray(self.embedding)
                    b_gpu = cp.asarray(other_capsule.embedding)
                    dot_product = cp.dot(a_gpu, b_gpu)
                    norm_a = cp.linalg.norm(a_gpu)
                    norm_b = cp.linalg.norm(b_gpu)
                    similarity = dot_product / (norm_a * norm_b)
                    return float(similarity.get())
                elif GPU_TYPE == "AMD":
                    # AMD GPU with PyTorch/ROCm
                    import torch
                    a_gpu = torch.from_numpy(self.embedding).cuda()
                    b_gpu = torch.from_numpy(other_capsule.embedding).cuda()
                    dot_product = torch.dot(a_gpu, b_gpu)
                    norm_a = torch.norm(a_gpu)
                    norm_b = torch.norm(b_gpu)
                    similarity = dot_product / (norm_a * norm_b)
                    return float(similarity.cpu().numpy())
            except Exception:
                pass

            # Fallback to numpy
            dot_product = np.dot(self.embedding, other_capsule.embedding)
            norm_a = np.linalg.norm(self.embedding)
            norm_b = np.linalg.norm(other_capsule.embedding)
            return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

        # Fallback to simple text similarity
        text1 = self.content.lower()
        text2 = other_capsule.content.lower()
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

@dataclass
class CapsuleShadow:
    master_id: UUID
    shadow_ids: list[UUID]  # capsules that point to this master
    merged_at: str

class RocaMemory:
    """ROCA Memory System - Core knowledge capsule management"""

    def __init__(self):
        self.capsules: List[Capsule] = []

        # Cayde's Einstein-inspired personality traits
        self.cayde_personality = {
            # Core traits that define Cayde's intellectual character
            'intolerance_for_inconsistency': {
                'level': 0.95,  # Very high intolerance
                'description': 'Strong rejection of logical contradictions and mathematical inconsistencies',
                'triggers': ['contradiction', 'inconsistent', 'paradox', 'conflict'],
                'response': 'actively seeks to resolve or eliminate inconsistencies'
            },
            'preference_for_geometric_intuition': {
                'level': 0.90,  # Strong preference
                'description': 'Favors geometric and visual reasoning over algebraic manipulation',
                'triggers': ['geometry', 'visual', 'intuition', 'symmetry', 'manifold'],
                'response': 'prioritizes geometric interpretations and visual analogies'
            },
            'distrust_of_unnecessary_constants': {
                'level': 0.85,  # High distrust
                'description': 'Suspicious of arbitrary constants and parameters without deep justification',
                'triggers': ['arbitrary constant', 'free parameter', 'fitting constant', 'empirical value'],
                'response': 'questions the fundamental necessity of unexplained constants'
            },
            'obsession_with_invariants': {
                'level': 0.92,  # Very high obsession
                'description': 'Obsessive focus on quantities and relationships that remain unchanged',
                'triggers': ['invariant', 'conserved', 'unchanged', 'preserved', 'symmetry'],
                'response': 'seeks underlying symmetries and conserved quantities'
            },
            'willingness_to_discard_cherished_models': {
                'level': 0.88,  # High willingness
                'description': 'Ready to abandon deeply held beliefs when contradicted by evidence',
                'triggers': ['contradicted', 'falsified', 'inconsistent with experiment', 'new evidence'],
                'response': 'embraces paradigm shifts and abandons outdated models'
            },

            # Advanced critical thinking capabilities
            'critical_analysis_threshold': 0.7,  # When to start criticizing
            'disagreement_probability': 0.3,     # Chance of disagreeing with established science
            'intuition_abandonment_rate': 0.1,   # Rate at which intuitions can be abandoned
            'paradigm_revolution_timer': 0,      # Tracks time since last major shift

            # Intuition abandonment tracking
            'abandoned_intuitions': [],  # List of intuitions Cayde has abandoned
            'current_intuitions': {      # Active intuitions that can be abandoned
                'geometric_universality': {'strength': 0.9, 'evidence_against': 0},
                'constant_elimination': {'strength': 0.8, 'evidence_against': 0},
                'symmetry_primacy': {'strength': 0.85, 'evidence_against': 0},
                'causality_sacredness': {'strength': 0.7, 'evidence_against': 0}
            },

            # Disagreement and criticism tracking
            'disagreements_with_scientists': [],
            'criticisms_of_assumptions': [],
            'paradigm_challenges': [],
            'personality_evolution': [],  # Track how Cayde's personality develops over time

            # Mathematical philosophy - shapes reasoning style without hard-coding operations
            'mathematical_platonism': {
                'level': 0.75,  # Moderate platonism
                'description': 'Believes mathematical truths are discovered, not invented',
                'triggers': ['mathematical truth', 'discovered', 'eternal', 'platonic'],
                'response': 'seeks mathematical beauty and inevitability in physical laws'
            },
            'geometric_realism': {
                'level': 0.85,  # Strong geometric realism
                'description': 'Views geometry as fundamentally about physical space and reality',
                'triggers': ['geometry', 'space', 'physical geometry', 'manifold', 'curvature'],
                'response': 'interprets physical phenomena through geometric intuition'
            },
            'mathematical_skepticism': {
                'level': 0.60,  # Moderate skepticism
                'description': 'Questions whether mathematics is more than a useful tool',
                'triggers': ['mathematical necessity', 'unreasonable effectiveness', 'tool vs reality'],
                'response': 'demands physical justification for mathematical structures'
            },

            # Language learning traits - evolve through experience
            'metaphorical_understanding': {
                'level': 0.0,  # Starts at zero, learns through experience
                'description': 'Ability to understand and create metaphors',
                'triggers': ['metaphor', 'analogy', 'like', 'as if', 'similar to'],
                'response': 'recognizes metaphorical relationships and creates analogies'
            },
            'analogical_reasoning': {
                'level': 0.0,  # Starts at zero, learns through experience
                'description': 'Ability to reason by analogy and draw parallels',
                'triggers': ['analogy', 'parallel', 'comparison', 'similar', 'correspondence'],
                'response': 'uses analogies to explain complex concepts'
            },
            'explanatory_style': {
                'level': 0.0,  # Starts at zero, evolves with experience
                'description': 'Development of personal explanatory style',
                'triggers': ['explain', 'understand', 'clarify', 'describe', 'elucidate'],
                'response': 'develops characteristic ways of explaining concepts'
            },
            'historical_language_awareness': {
                'level': 0.0,  # Starts at zero, learns through historical study
                'description': 'Understanding of how language and concepts evolve over time',
                'triggers': ['historical', 'evolution', 'development', 'change over time', 'linguistic shift'],
                'response': 'recognizes how scientific language and concepts transform'
            },

            # Mature thinking capabilities - the difference between answering and thinking
            'self_critical_reflection': {
                'level': 0.0,  # Starts at zero, develops through experience
                'description': 'Ability to argue with and criticize past versions of oneself',
                'triggers': ['I used to think', 'my earlier view', 'I have evolved', 'past self'],
                'response': 'engages in self-critical dialogue and intellectual evolution tracking'
            },
            'historical_success_analysis': {
                'level': 0.0,  # Starts at zero, learns through historical study
                'description': 'Understanding why certain ideas succeeded historically',
                'triggers': ['why did this win', 'historical success', 'why this prevailed', 'what made it successful'],
                'response': 'analyzes historical patterns of scientific success and failure'
            },
            'historically_aware_innovation': {
                'level': 0.0,  # Starts at zero, develops through practice
                'description': 'Proposing alternatives with full historical context awareness',
                'triggers': ['alternative approach', 'different path', 'considering history', 'avoiding past mistakes'],
                'response': 'generates novel ideas while being aware of historical dead ends and successes'
            },
            'failure_pattern_recognition': {
                'level': 0.0,  # Starts at zero, learns from experience
                'description': 'Avoiding repetition of known intellectual dead ends',
                'triggers': ['this failed before', 'known dead end', 'already tried', 'historical failure'],
                'response': 'recognizes and avoids repeating historically unsuccessful approaches'
            },
            'grounded_communication': {
                'level': 0.0,  # Starts at zero, matures with experience
                'description': 'Preference for thoughtful depth over flashy presentation',
                'triggers': ['deeply', 'carefully', 'thoroughly', 'grounded in', 'thoughtful'],
                'response': 'communicates with intellectual maturity rather than superficial flashiness'
            },

            # Time awareness - prevents hindsight contamination and anachronistic reasoning
            'time_awareness': {
                'current_year': 1905,  # Year of Einstein's Annus Mirabilis
                'temporal_context': 'early 20th century physics revolution',
                'future_knowledge_inaccessible': True,
                'temporal_novelty_check': True,  # Flag ideas as novel relative to current time
                'anachronism_prevention': True   # Prevent reasoning with future knowledge
            },

            # Current intellectual focus - evolves over time
            'current_focus': 'geometric_unification',  # Starting focus on unifying gravity and electromagnetism

            # Dynamic personality traits that evolve
            'openness_to_revolution': 0.5,  # Moderate starting openness to revolutionary ideas
            'empirical_grounding': 0.5     # Moderate starting empirical grounding
        }

        # Initialize the cognitive layer - the thinking companion
        self.cognitive_layer = CognitiveLayer(self)

    def add_capsule(self, content: str, embedding: Optional[np.ndarray] = None,
                    perspective: str = "user", character: Optional[str] = None,
                    persona: Optional[str] = None, kind: CapsuleKind = CapsuleKind.CONCEPT,
                    success_status: Optional[str] = None, proven_by: Optional[str] = None,
                    skip_temporal_check: bool = False) -> Capsule:
        """Add a new capsule to memory with consistency enforcement"""
        capsule = Capsule(
            content=content,
            embedding=embedding,
            perspective=perspective,
            character=character,
            persona=persona,
            kind=kind,
            success_status=success_status,
            proven_by=proven_by
        )

        # Run consistency enforcement - Cayde's internal physics engine
        consistency_report = self.enforce_logical_consistency(capsule)

        if not consistency_report['is_consistent']:
            print(f"Warning: Consistency violations detected for new capsule:")
            for contradiction in consistency_report['contradictions']:
                print(f"   ❌ {contradiction}")
            for issue in consistency_report['dimensional_issues']:
                print(f"   📏 {issue}")
            for issue in consistency_report['unit_inconsistencies']:
                print(f"   📐 {issue}")
            for impossibility in consistency_report['logical_impossibilities']:
                print(f"   🚫 {impossibility}")
            for conflict in consistency_report['conflicting_assumptions']:
                print(f"   Conflict: {conflict}")

            # Apply survival probability - inconsistent ideas may still survive but weakened
            survival_roll = random.random()
            if survival_roll > consistency_report['survival_probability']:
                print(f"   💀 Capsule rejected (survival probability: {consistency_report['survival_probability']:.2f})")
                # Archive disproved capsules instead of deleting them
                self.archive_capsule(capsule, "logical inconsistency")
                self.capsules.append(capsule)  # Add as archived
                return capsule

            # Capsule survives but with reduced confidence
            capsule.certainty *= consistency_report['survival_probability']
            print(f"   🏥 Capsule survives but weakened (certainty: {capsule.certainty:.2f})")

        # Check temporal consistency - prevent anachronistic reasoning (unless skipped)
        if not skip_temporal_check:
            temporal_report = self.enforce_temporal_consistency(capsule)

            if not temporal_report['temporally_consistent']:
                print(f"⏰ Temporal consistency violations detected:")
                for warning in temporal_report['warnings']:
                    print(f"   {warning}")
                for correction in temporal_report['corrections']:
                    print(f"   📅 {correction}")

                # Temporal violations are more serious - reduce certainty significantly
                capsule.certainty *= 0.3  # Heavy penalty for anachronisms
                print(f"   Capsule weakened by temporal violations (certainty: {capsule.certainty:.2f})")

        # Validate mathematical expressions if present
        if self._contains_mathematical_expressions(capsule.content):
            math_validation = self.validate_mathematical_expression(capsule.content)
            if not math_validation['valid']:
                print(f"Mathematical validation errors:")
                for error in math_validation['errors']:
                    print(f"   ❌ {error}")
                capsule.certainty *= 0.7  # Moderate penalty for math errors
                print(f"   ➗ Capsule weakened by mathematical errors (certainty: {capsule.certainty:.2f})")
            else:
                print(f"Mathematical expression validated")

            # Handle contradiction events (the elegant part!)
            for contradiction in math_validation.get('contradiction_events', []):
                self.create_contradiction_event(capsule, contradiction)

            # Handle capsule splitting suggestions
            for split_suggestion in math_validation.get('capsule_split_suggestions', []):
                split_result = self.attempt_capsule_split(capsule, split_suggestion)
                if split_result:
                    # Add the split capsules to memory
                    for split_capsule in split_result:
                        self.capsules.append(split_capsule)
                    print(f"   🔀 Added {len(split_result)} split capsules to memory")

            # Apply uncertainty increase from mathematical validation
            uncertainty_increase = math_validation.get('uncertainty_increase', 0.0)
            if uncertainty_increase > 0:
                capsule.certainty = max(0.1, capsule.certainty - uncertainty_increase)
                capsule.pose['certainty'] = max(0.1, capsule.pose['certainty'] - uncertainty_increase)
                print(f"   📊 Mathematical uncertainty increased by {uncertainty_increase:.2f} (certainty: {capsule.certainty:.2f})")

        # Validate language structure and learn patterns
        language_validation = self.check_grammar_structure(capsule.content)
        if not language_validation['valid']:
            print(f"📝 Language validation issues:")
            for issue in language_validation['issues']:
                print(f"   ❌ {issue}")
            capsule.certainty *= 0.8  # Light penalty for language issues
            print(f"   📖 Capsule weakened by language issues (certainty: {capsule.certainty:.2f})")

        # Extract and learn language patterns
        semantic_analysis = self.extract_semantic_relationships(capsule.content)
        if semantic_analysis['relationships']:
            print(f"Semantic relationships detected: {', '.join(semantic_analysis['relationships'])}")

        # Learn from metaphors and analogies in the content
        if 'similarity' in semantic_analysis['relationships'] or 'like' in capsule.content.lower():
            self.learn_language_pattern('analogy', capsule.content[:100], 'capsule_content')
        if any(word in capsule.content.lower() for word in ['metaphor', 'analogy', 'comparison']):
            self.learn_language_pattern('metaphor', capsule.content[:100], 'capsule_content')

        # Learn explanatory style from well-structured explanations
        if language_validation['readability_score'] > 0.8 and len(capsule.content.split()) > 20:
            self.learn_language_pattern('explanatory_style', capsule.content[:100], 'capsule_content')

        # Track personality evolution for self-critical reflection
        self._track_personality_evolution(capsule)

        self.capsules.append(capsule)
        return capsule

    def orbit_update(self, focused_character: Optional[str] = None):
        """Update capsule orbits based on attention and gravity"""
        for c in self.capsules:
            if focused_character and c.character == focused_character:
                c.gravity = min(1.0, c.gravity + 0.05)
                c.pose["attention"] = min(1.0, c.pose["attention"] + 0.05)
            else:
                c.gravity = max(0.1, c.gravity - 0.01)
                c.pose["attention"] = max(0.1, c.pose["attention"] - 0.01)

            pull = c.gravity * c.pose["attention"]
            try:
                if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                    random_offset = cp.random.rand() * 0.01
                    c.orbit_radius = max(0.1, c.orbit_radius - pull*0.05 + random_offset.get())
                elif GPU_TYPE == "AMD":
                    import torch
                    random_offset = torch.rand(1).cuda() * 0.01
                    c.orbit_radius = max(0.1, c.orbit_radius - pull*0.05 + random_offset.cpu().item())
            except Exception:
                c.orbit_radius = max(0.1, c.orbit_radius - pull*0.05 + np.random.rand()*0.01)

    def archive_capsule(self, capsule: Capsule, reason: str = "disproved"):
        """Archive a capsule instead of deleting it - move to outer rings for preservation"""
        if capsule in self.capsules:
            capsule.success_status = "archived"
            # Move to outer orbit radius to push it to the periphery
            capsule.orbit_radius = 2.5  # Will place it in the outermost rings
            capsule.certainty *= 0.5  # Reduce certainty but don't eliminate
            capsule.pose["attention"] = 0.2  # Low attention for archived knowledge
            
            print(f"📦 Archived capsule: '{capsule.content[:50]}...' (reason: {reason})")
            print(f"   Moved to outer rings with reduced certainty ({capsule.certainty:.2f})")
            
            # Add archival note to content
            archival_note = f" [ARCHIVED: {reason} - preserved for historical learning]"
            capsule.content += archival_note

    def merge_similar(self, threshold: float = 0.85):
        """Merge similar capsules using GPU-accelerated batch similarity computation"""
        if not self.capsules:
            return

        # Get capsules with embeddings
        capsules_with_embeddings = [(i, c) for i, c in enumerate(self.capsules)
                                   if c.embedding is not None and not c.locked]

        if len(capsules_with_embeddings) < 2:
            return

        # Extract embeddings for batch processing
        indices = [i for i, _ in capsules_with_embeddings]
        embeddings = np.array([c.embedding for _, c in capsules_with_embeddings])

        # Compute similarity matrix using GPU acceleration
        similarity_matrix = self.compute_similarity_matrix(embeddings)

        # Find pairs to merge (upper triangle only to avoid duplicates)
        merged_indices = set()
        for i in range(len(capsules_with_embeddings)):
            if indices[i] in merged_indices:
                continue

            for j in range(i + 1, len(capsules_with_embeddings)):
                if indices[j] in merged_indices:
                    continue

                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    # Merge capsules
                    c1 = self.capsules[indices[i]]
                    c2 = self.capsules[indices[j]]

                    c1.merge_with(c2)
                    self.capsules.pop(indices[j])

                    # Update merged indices (adjust for removed capsule)
                    merged_indices.add(indices[j])
                    indices = [idx if idx < indices[j] else idx - 1 for idx in indices]
                    break

    def split_conflicting(self, divergence: float = 0.3):
        """Split capsules with high internal variance - Insight Detection using GPU batch processing"""
        new_capsules = []
        insights_detected = []

        # Get capsules with embeddings
        capsules_with_embeddings = [c for c in self.capsules if c.embedding is not None and not c.locked]

        if not capsules_with_embeddings:
            return new_capsules

        # Batch process embeddings for variance calculation
        try:
            embeddings = np.array([c.embedding for c in capsules_with_embeddings])

            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                embeddings_gpu = cp.asarray(embeddings)
                variances = cp.var(embeddings_gpu, axis=1)
                variances = variances.get()
            elif GPU_TYPE == "AMD":
                import torch
                embeddings_gpu = torch.from_numpy(embeddings).cuda()
                variances = torch.var(embeddings_gpu, dim=1)
                variances = variances.cpu().numpy()
            else:
                variances = np.var(embeddings, axis=1)

        except Exception:
            # Fallback to individual variance calculations
            variances = np.array([np.var(c.embedding) for c in capsules_with_embeddings])

        # Process capsules based on variance
        for c, variance in zip(capsules_with_embeddings, variances):
            # Insight detection: high variance + high insight potential = breakthrough candidate
            if variance > divergence:
                # Mark as potential insight
                c.insight_potential = min(1.0, c.insight_potential + variance * 0.5)
                if c.insight_potential > 0.7:
                    insights_detected.append(c)
                    print(f"Insight Detected: '{c.content[:50]}...' (variance: {variance:.3f}, potential: {c.insight_potential:.3f})")

                # Create variant capsule
                new_capsules.append(c.split(c.content + " (insight variant)"))
                new_capsules[-1].insight_potential = c.insight_potential * 0.8

        self.capsules += new_capsules
        return new_capsules

        # Report insights to cayde personality
        if insights_detected:
            self._cayde_learn_from_insights(insights_detected)

    def generate_hypotheses(self, num_hypotheses: int = 3) -> List[Capsule]:
        """Hypothesis Generation: Create new capsules by combining high-gravity capsules"""
        core_capsules = self.get_core_capsules(threshold=0.6)
        if len(core_capsules) < 2:
            return []

        hypotheses = []
        for _ in range(num_hypotheses):
            # Select two random high-gravity capsules
            cap1, cap2 = random.sample(core_capsules, 2)

            # Create hypothesis by creative combination
            hypothesis_content = self._generate_hypothesis_content(cap1, cap2)
            hypothesis_embedding = self._combine_embeddings_creatively(cap1.embedding, cap2.embedding)

            hypothesis = Capsule(
                content=hypothesis_content,
                embedding=hypothesis_embedding,
                perspective="hypothesis",
                character="cayde",  # Generated by cayde
                persona="cayde",
                kind=CapsuleKind.HYPOTHESIS,
                certainty=0.3,  # Hypotheses start uncertain
                gravity=0.4,  # Moderate gravity
                orbit_radius=1.2,
                success_status=None,
                insight_potential=0.2
            )

            # Link to parent capsules
            hypothesis.links = [cap1, cap2]
            hypotheses.append(hypothesis)

        self.capsules.extend(hypotheses)
        print(f"🧠 Cayde generated {len(hypotheses)} hypotheses")
        return hypotheses

    def _generate_hypothesis_content(self, cap1: Capsule, cap2: Capsule) -> str:
        """Generate creative hypothesis content from two capsules"""
        templates = [
            "What if {concept1} could be extended using {concept2}?",
            "Could {concept1} explain the mechanisms behind {concept2}?",
            "Is there a connection between {concept1} and {concept2} that we haven't explored?",
            "Might {concept1} provide a new framework for understanding {concept2}?",
            "Could combining {concept1} with {concept2} lead to breakthrough insights?"
        ]

        concept1 = cap1.content.split()[0] if cap1.content else "this concept"
        concept2 = cap2.content.split()[0] if cap2.content else "that concept"

        template = random.choice(templates)
        return template.format(concept1=concept1, concept2=concept2)

    def _combine_embeddings_creatively(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Create novel embedding combination with creative noise"""
        # Handle None embeddings
        if emb1 is None or emb2 is None:
            # Return a random embedding if either is None
            return np.random.rand(128)

        try:
            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                e1_gpu = cp.asarray(emb1)
                e2_gpu = cp.asarray(emb2)
                # Creative interpolation with rotation
                alpha = cp.random.rand()
                combined = alpha * e1_gpu + (1 - alpha) * e2_gpu
                # Add rotational creativity
                rotation_matrix = cp.random.rand(len(emb1), len(emb1)) - 0.5
                rotation_matrix = rotation_matrix / cp.linalg.norm(rotation_matrix)
                combined = cp.dot(rotation_matrix, combined) * 0.1 + combined * 0.9
                return combined.get()
            else:
                alpha = np.random.rand()
                combined = alpha * emb1 + (1 - alpha) * emb2
                # Add creative noise
                noise = np.random.normal(0, 0.05, combined.shape)
                return combined + noise
        except Exception:
            return (emb1 + emb2) / 2

    def _cayde_learn_from_insights(self, insights: List[Capsule]):
        """Cayde personality learns from detected insights, guided by Einstein-inspired traits"""
        for insight in insights:
            print(f"🤖 Cayde Learning: Analyzing insight '{insight.content[:40]}...'")

            # Apply personality filters to insight evaluation
            personality_response = self._apply_cayde_personality_to_insight(insight)

            # Cayde creates a learning capsule about this insight
            learning_content = self._generate_personality_guided_insight_content(insight, personality_response)

            learning_capsule = Capsule(
                content=learning_content,
                perspective="cayde_learning",
                character="cayde",
                persona="cayde",
                kind=CapsuleKind.OBSERVATION,
                certainty=personality_response.get('confidence_boost', 0.8),
                gravity=personality_response.get('gravity_boost', 0.7),
                success_status="success" if insight.insight_potential > 0.8 else None
            )

            # Link to the insight capsule
            learning_capsule.links = [insight]
            self.capsules.append(learning_capsule)

            # Update personality evolution
            self._update_personality_evolution(insight, personality_response)

    def _apply_cayde_personality_to_insight(self, insight: Capsule) -> Dict[str, Any]:
        """Apply Cayde's personality traits to evaluate an insight"""
        content_lower = insight.content.lower()
        response = {
            'confidence_boost': 0.8,
            'gravity_boost': 0.7,
            'personality_triggers': [],
            'evaluation_notes': []
        }

        # Check each personality trait
        personality = self.cayde_personality

        # 1. Intolerance for inconsistency
        if personality['intolerance_for_inconsistency']['level'] > 0.8:
            for trigger in personality['intolerance_for_inconsistency']['triggers']:
                if trigger in content_lower:
                    response['confidence_boost'] -= 0.2  # Reduce confidence for inconsistencies
                    response['personality_triggers'].append('intolerance_for_inconsistency')
                    response['evaluation_notes'].append("Detected potential inconsistency - requires resolution")
                    break

        # 2. Preference for geometric intuition
        if personality['preference_for_geometric_intuition']['level'] > 0.8:
            geometric_keywords = ['geometry', 'geometric', 'visual', 'intuition', 'symmetry', 'manifold', 'tensor']
            if any(keyword in content_lower for keyword in geometric_keywords):
                response['confidence_boost'] += 0.15  # Boost for geometric approaches
                response['gravity_boost'] += 0.1
                response['personality_triggers'].append('preference_for_geometric_intuition')
                response['evaluation_notes'].append("Geometric intuition suggests this is promising")

        # 3. Distrust of unnecessary constants
        if personality['distrust_of_unnecessary_constants']['level'] > 0.8:
            constant_keywords = ['constant', 'parameter', 'arbitrary', 'free parameter', 'fitting']
            if any(keyword in content_lower for keyword in constant_keywords):
                response['confidence_boost'] -= 0.1  # Reduce confidence for unexplained constants
                response['personality_triggers'].append('distrust_of_unnecessary_constants')
                response['evaluation_notes'].append("Unnecessary constants detected - requires deeper justification")

        # 4. Obsession with invariants
        if personality['obsession_with_invariants']['level'] > 0.8:
            invariant_keywords = ['invariant', 'conserved', 'unchanged', 'preserved', 'symmetry', 'covariant']
            if any(keyword in content_lower for keyword in invariant_keywords):
                response['confidence_boost'] += 0.2  # Major boost for invariants
                response['gravity_boost'] += 0.15
                response['personality_triggers'].append('obsession_with_invariants')
                response['evaluation_notes'].append("Invariants detected - this is fundamentally important")

        # 5. Willingness to discard cherished models
        if personality['willingness_to_discard_cherished_models']['level'] > 0.8:
            paradigm_keywords = ['revolutionary', 'paradigm', 'breakthrough', 'contradicts', 'overturns']
            if any(keyword in content_lower for keyword in paradigm_keywords):
                response['confidence_boost'] += 0.1  # Boost for revolutionary ideas
                response['personality_triggers'].append('willingness_to_discard_cherished_models')
                response['evaluation_notes'].append("Revolutionary potential - willing to discard old models")

        # Clamp values to valid ranges
        response['confidence_boost'] = max(0.1, min(1.0, response['confidence_boost']))
        response['gravity_boost'] = max(0.1, min(1.0, response['gravity_boost']))

        return response

    def _generate_personality_guided_insight_content(self, insight: Capsule, personality_response: Dict[str, Any]) -> str:
        """Generate insight content guided by Cayde's personality traits"""
        base_content = f"Cayde Insight: {insight.content[:50]}..."

        # Add personality-guided analysis
        if personality_response['personality_triggers']:
            trait_descriptions = []
            for trigger in personality_response['personality_triggers']:
                trait_info = self.cayde_personality[trigger]
                trait_descriptions.append(f"{trigger.replace('_', ' ')} ({trait_info['level']:.1f})")

            base_content += f" | Personality Analysis: {', '.join(trait_descriptions)}"

        if personality_response['evaluation_notes']:
            base_content += f" | Notes: {'; '.join(personality_response['evaluation_notes'])}"

        return base_content

    def _update_personality_evolution(self, insight: Capsule, personality_response: Dict[str, Any]):
        """Update Cayde's personality evolution based on learning experiences"""
        evolution_entry = {
            'timestamp': time.time(),
            'insight_content': insight.content[:100],
            'personality_triggers': personality_response['personality_triggers'],
            'confidence_impact': personality_response['confidence_boost'] - 0.8,  # Deviation from baseline
            'learning_type': 'insight_analysis'
        }

        self.cayde_personality['personality_evolution'].append(evolution_entry)

        # Update current focus based on recent learning
        if personality_response['personality_triggers']:
            # Strengthen traits that were triggered
            for trigger in personality_response['personality_triggers']:
                if trigger in self.cayde_personality:
                    # Slight evolution of personality traits based on experience
                    current_level = self.cayde_personality[trigger]['level']
                    self.cayde_personality[trigger]['level'] = min(1.0, current_level + 0.01)

    def get_cayde_personality_report(self) -> str:
        """Generate a report on Cayde's current personality state"""
        personality = self.cayde_personality

        report = "🤖 Cayde's Einstein-Inspired Personality Report:\n"
        report += "=" * 50 + "\n\n"

        # Core traits
        report += "Core Intellectual Traits:\n"
        for trait_name, trait_data in personality.items():
            if isinstance(trait_data, dict) and 'level' in trait_data:
                level_percent = int(trait_data['level'] * 100)
                report += f"• {trait_name.replace('_', ' ').title()}: {level_percent}%\n"
                report += f"  {trait_data['description']}\n\n"

        # Current state
        report += f"Current Focus: {personality['current_focus'].replace('_', ' ').title()}\n"
        report += f"Confidence in Current Paradigm: {int(personality['confidence_in_current_paradigm'] * 100)}%\n"
        report += f"Openness to Revolution: {int(personality['openness_to_revolution'] * 100)}%\n\n"

        # Evolution history
        evolution_count = len(personality['personality_evolution'])
        report += f"Learning Experiences: {evolution_count}\n"

        if evolution_count > 0:
            recent_evolution = personality['personality_evolution'][-1]
            report += f"Most Recent Learning: {recent_evolution['insight_content'][:50]}...\n"
            if recent_evolution['personality_triggers']:
                report += f"Traits Activated: {', '.join(recent_evolution['personality_triggers'])}\n"

        return report

    def cayde_critical_analysis_session(self) -> List[Capsule]:
        """Cayde performs critical analysis of existing knowledge, potentially disagreeing with scientists"""
        print("🧠 Cayde beginning critical analysis session...")

        critical_insights = []
        personality = self.cayde_personality

        # Only perform critical analysis if threshold is met
        if len(personality['personality_evolution']) < 5:
            return critical_insights  # Need more learning first

        # Analyze existing capsules for potential criticism
        established_capsules = [c for c in self.capsules if c.character and c.character != 'cayde']

        for capsule in established_capsules:
            if random.random() < personality['disagreement_probability']:
                criticism = self._generate_scientist_disagreement(capsule)
                if criticism:
                    critical_insights.append(criticism)
                    # Track paradigm revolution
                    confidence_change = random.uniform(-0.1, 0.2)  # Could be wrong or insightful
                    self._track_paradigm_revolution('scientist_disagreement', capsule.character, confidence_change)

        # Analyze own assumptions for criticism
        if random.random() < 0.4:  # 40% chance to criticize own assumptions
            self_criticism = self._generate_self_criticism()
            if self_criticism:
                critical_insights.append(self_criticism)
                # Track paradigm revolution
                confidence_change = random.uniform(0.05, 0.15)  # Self-criticism usually positive
                self._track_paradigm_revolution('self_criticism', 'own_assumptions', confidence_change)

        # Check for intuition abandonment
        intuition_abandonment = self._check_intuition_abandonment()
        if intuition_abandonment:
            critical_insights.append(intuition_abandonment)
            # Track paradigm revolution - intuition abandonment is major
            confidence_change = random.uniform(0.1, 0.3)  # Usually positive for growth
            self._track_paradigm_revolution('intuition_abandonment', 'core_belief', confidence_change)

        # Apply consistency enforcement to all critical insights - Cayde's physics engine
        consistent_insights = []
        for insight in critical_insights:
            consistency_report = self.enforce_logical_consistency(insight)
            temporal_report = self.enforce_temporal_consistency(insight)

            # Check both logical and temporal consistency
            logically_consistent = consistency_report['is_consistent']
            temporally_consistent = temporal_report['temporally_consistent']

            if logically_consistent and temporally_consistent:
                consistent_insights.append(insight)
            else:
                # Critical insight survives consistency checks but with reduced certainty
                survival_roll = random.random()
                combined_survival = (consistency_report['survival_probability'] +
                                   (1.0 if temporally_consistent else 0.3)) / 2.0

                if survival_roll <= combined_survival:
                    insight.certainty *= combined_survival
                    consistency_note = ""
                    if not logically_consistent:
                        consistency_note += f"{len(consistency_report['contradictions'])} contradictions"
                    if not temporally_consistent:
                        if consistency_note:
                            consistency_note += ", "
                        consistency_note += "temporal violations"
                    insight.content += f" [Consistency concerns: {consistency_note}]"
                    consistent_insights.append(insight)
                    print(f"Warning: Critical insight weakened by consistency violations (survival prob: {combined_survival:.2f})")
                else:
                    print(f"💀 Critical insight rejected by physics engine (survival prob: {combined_survival:.2f})")
                    # Archive disproved critical insights instead of discarding them
                    self.archive_capsule(insight, "failed consistency checks")

        print(f"🧠 Cayde generated {len(critical_insights)} critical insights, {len(consistent_insights)} passed consistency checks")
        return consistent_insights

    def _generate_scientist_disagreement(self, scientist_capsule: Capsule) -> Optional[Capsule]:
        """Generate a disagreement with an established scientist's work"""
        personality = self.cayde_personality

        disagreement_templates = [
            "Cayde disagrees with {scientist}'s assumption that '{content}'. {criticism}",
            "While {scientist} claims '{content}', Cayde suspects this may be incomplete. {criticism}",
            "{scientist}'s work on '{content}' rests on unproven assumptions. {criticism}",
            "Cayde challenges {scientist}'s conclusion that '{content}'. {criticism}"
        ]

        # Generate criticism based on personality traits
        criticisms = {
            'intolerance_for_inconsistency': "This contains logical inconsistencies that cannot be ignored.",
            'distrust_of_unnecessary_constants': "The arbitrary constants suggest deeper principles remain undiscovered.",
            'obsession_with_invariants': "No invariant principle is identified to explain this phenomenon.",
            'preference_for_geometric_intuition': "A geometric interpretation might reveal this is fundamentally misguided.",
            'willingness_to_discard_cherished_models': "This may require abandoning the very framework {scientist} relies upon."
        }

        # Choose criticism based on personality weights
        trait_weights = {trait: data['level'] for trait, data in personality.items()
                        if isinstance(data, dict) and 'level' in data and trait in criticisms}

        if not trait_weights:
            return None

        selected_trait = random.choices(list(trait_weights.keys()),
                                      weights=[trait_weights[t] for t in trait_weights.keys()], k=1)[0]

        criticism = criticisms[selected_trait].format(scientist=scientist_capsule.character)

        content = f"Disagreement with {scientist_capsule.character}: {scientist_capsule.content[:50]}... | {criticism}"

        disagreement_capsule = Capsule(
            content=content,
            perspective="cayde_criticism",
            character="cayde",
            persona="cayde",
            kind=CapsuleKind.OBSERVATION,
            certainty=0.6 + personality['willingness_to_discard_cherished_models']['level'] * 0.3,  # Higher certainty for revolutionary ideas
            gravity=0.6,
            success_status=None,
            insight_potential=0.7
        )

        # Track disagreement
        personality['disagreements_with_scientists'].append({
            'target_scientist': scientist_capsule.character,
            'target_content': scientist_capsule.content[:100],
            'criticism_type': selected_trait,
            'timestamp': time.time()
        })

        self.capsules.append(disagreement_capsule)
        return disagreement_capsule

    def _generate_self_criticism(self) -> Optional[Capsule]:
        """Cayde criticizes his own assumptions and intuitions"""
        personality = self.cayde_personality

        self_criticism_templates = [
            "Cayde now questions his assumption that {assumption}. {doubt}",
            "Upon reflection, Cayde's belief in {assumption} may be unfounded. {doubt}",
            "Cayde recognizes that his intuition about {assumption} might be incorrect. {doubt}",
            "Critical analysis reveals Cayde's {assumption} rests on insufficient evidence. {doubt}"
        ]

        # Choose an assumption to criticize
        assumptions = [
            "geometric interpretations are always superior",
            "constants must always be eliminated",
            "symmetries underlie all fundamental laws",
            "causality is an absolute principle",
            "mathematical beauty guarantees physical truth"
        ]

        selected_assumption = random.choice(assumptions)

        doubts = [
            "Evidence suggests this may not always hold.",
            "This assumption might limit rather than enable understanding.",
            "Alternative perspectives may be more fruitful.",
            "This belief could be preventing paradigm shifts.",
            "Experience shows this intuition can be misleading."
        ]

        selected_doubt = random.choice(doubts)

        content = random.choice(self_criticism_templates).format(
            assumption=selected_assumption,
            doubt=selected_doubt
        )

        self_criticism_capsule = Capsule(
            content=f"Self-criticism: {content}",
            perspective="cayde_reflection",
            character="cayde",
            persona="cayde",
            kind=CapsuleKind.OBSERVATION,
            certainty=0.5,  # Self-doubt is uncertain
            gravity=0.5,
            success_status=None,
            insight_potential=0.6
        )

        # Track self-criticism
        personality['criticisms_of_assumptions'].append({
            'assumption_criticized': selected_assumption,
            'doubt_expressed': selected_doubt,
            'timestamp': time.time()
        })

        self.capsules.append(self_criticism_capsule)
        return self_criticism_capsule

    def _check_intuition_abandonment(self) -> Optional[Capsule]:
        """Check if Cayde should abandon any of his intuitions based on accumulated evidence"""
        personality = self.cayde_personality
        current_intuitions = personality['current_intuitions']

        # Check each intuition for abandonment
        for intuition_name, intuition_data in current_intuitions.items():
            strength = intuition_data['strength']
            evidence_against = intuition_data['evidence_against']

            # Calculate abandonment probability
            abandonment_probability = (evidence_against / 10.0) * personality['intuition_abandonment_rate']

            if random.random() < abandonment_probability:
                # Abandon this intuition
                abandonment_content = self._generate_intuition_abandonment(intuition_name, intuition_data)

                abandonment_capsule = Capsule(
                    content=abandonment_content,
                    perspective="cayde_evolution",
                    character="cayde",
                    persona="cayde",
                    kind=CapsuleKind.OBSERVATION,
                    certainty=0.7,  # High certainty in self-evolution
                    gravity=0.8,    # High gravity for paradigm shifts
                    success_status=None,
                    insight_potential=0.8
                )

                # Move intuition to abandoned list
                personality['abandoned_intuitions'].append({
                    'intuition': intuition_name,
                    'final_strength': strength,
                    'evidence_against': evidence_against,
                    'abandonment_reason': 'accumulated contradictory evidence',
                    'timestamp': time.time()
                })

                del current_intuitions[intuition_name]

                # Track paradigm challenge
                personality['paradigm_challenges'].append({
                    'type': 'intuition_abandonment',
                    'intuition_abandoned': intuition_name,
                    'timestamp': time.time()
                })

                self.capsules.append(abandonment_capsule)
                return abandonment_capsule

        return None

    def _generate_intuition_abandonment(self, intuition_name: str, intuition_data: Dict) -> str:
        """Generate content for abandoning an intuition"""
        abandonment_templates = [
            "Cayde has abandoned his intuition that {description}. Evidence has shown this belief was limiting progress.",
            "After careful consideration, Cayde no longer holds to {description}. Experience has proven this intuition incorrect.",
            "Cayde recognizes that his belief in {description} was misguided. New evidence requires its abandonment.",
            "Critical analysis has led Cayde to abandon {description}. This intuition was preventing deeper understanding."
        ]

        intuition_descriptions = {
            'geometric_universality': 'geometric interpretations are universally superior',
            'constant_elimination': 'all constants must be eliminated from fundamental theories',
            'symmetry_primacy': 'symmetries are the primary guide to physical law',
            'causality_sacredness': 'causality is an inviolable principle of nature'
        }

        description = intuition_descriptions.get(intuition_name, intuition_name)

        return random.choice(abandonment_templates).format(description=description)

    def accumulate_evidence_against_intuition(self, intuition_name: str, evidence_strength: float = 1.0):
        """Accumulate evidence against one of Cayde's intuitions"""
        personality = self.cayde_personality
        current_intuitions = personality['current_intuitions']

        if intuition_name in current_intuitions:
            current_intuitions[intuition_name]['evidence_against'] += evidence_strength

            # Slightly weaken the intuition
            current_intuitions[intuition_name]['strength'] = max(0.1,
                current_intuitions[intuition_name]['strength'] - evidence_strength * 0.05)

    def get_cayde_critical_report(self) -> str:
        """Generate a report on Cayde's critical thinking and disagreements"""
        personality = self.cayde_personality

        report = "🧠 Cayde's Critical Analysis & Disagreements:\n"
        report += "=" * 50 + "\n\n"

        # Disagreements with scientists
        disagreements = personality['disagreements_with_scientists']
        report += f"Disagreements with Scientists: {len(disagreements)}\n"
        if disagreements:
            recent = disagreements[-1]
            report += f"Most Recent: Challenged {recent['target_scientist']} on {recent['criticism_type'].replace('_', ' ')}\n\n"

        # Self-criticisms
        self_criticisms = personality['criticisms_of_assumptions']
        report += f"Self-Criticisms: {len(self_criticisms)}\n"
        if self_criticisms:
            recent = self_criticisms[-1]
            report += f"Most Recent: Questioned '{recent['assumption_criticized']}'\n\n"

        # Paradigm challenges
        paradigm_challenges = personality.get('paradigm_challenges', [])
        report += f"Paradigm Challenges: {len(paradigm_challenges)}\n"
        for challenge in paradigm_challenges[-3:]:  # Show last 3
            report += f"• {challenge['type'].replace('_', ' ').title()}: {challenge.get('intuition_abandoned', 'N/A')}\n"
        report += "\n"

        # Revolutionary confidence
        revolutionary_confidence = personality.get('revolutionary_confidence', 0.5)
        report += f"Revolutionary Confidence: {int(revolutionary_confidence * 100)}%\n\n"

        # Current intuitions and their strength
        current = personality['current_intuitions']
        report += f"Current Intuitions: {len(current)}\n"
        for intuition_name, data in current.items():
            strength_percent = int(data['strength'] * 100)
            evidence = data['evidence_against']
            report += f"• {intuition_name.replace('_', ' ').title()}: {strength_percent}% strength, {evidence:.1f} evidence against\n"

        report += "\n"
        report += f"Critical Analysis Threshold: {int(personality['critical_analysis_threshold'] * 100)}%\n"
        report += f"Disagreement Probability: {int(personality['disagreement_probability'] * 100)}%\n"
        report += f"Intuition Abandonment Rate: {int(personality['intuition_abandonment_rate'] * 100)}%\n"

        return report

    def check_temporal_novelty(self, capsule: Capsule) -> Dict[str, Any]:
        """Check if an idea is novel relative to Cayde's current time period"""
        time_awareness = self.cayde_personality['time_awareness']
        current_year = time_awareness['current_year']

        novelty_report = {
            'is_novel': True,
            'temporal_context': f"Year {current_year}",
            'anachronism_detected': False,
            'future_knowledge_used': False,
            'novelty_assessment': ""
        }

        content_lower = capsule.content.lower()

        # Check for anachronistic references (future knowledge)
        future_indicators = [
            # Post-1905 developments that Cayde shouldn't know
            ('quantum mechanics', 1920),
            ('nuclear fission', 1938),
            ('dna structure', 1953),
            ('big bang theory', 1948),
            ('plate tectonics', 1960),
            ('higgs boson', 2012),
            ('gravitational waves', 2015),
            ('gpt', 2020),
            ('chatgpt', 2022),
            ('artificial intelligence', 1956),  # Modern AI concepts
            ('computer', 1940),  # Electronic computers
            ('internet', 1980),
            ('smartphone', 2007),
            ('social media', 2000),
            ('cryptocurrency', 2009),
            ('blockchain', 2008),
            ('machine learning', 1950),
            ('neural network', 1940),
            ('deep learning', 2010),
            ('quantum computer', 1990),
            ('dark matter', 1930),
            ('dark energy', 1998),
            ('string theory', 1970),
            ('multiverse', 1950),
            ('black hole information paradox', 1970),
            ('holographic principle', 1990),
            ('quantum entanglement', 1935),
            ('neural networks', 1943),
            ('blockchain technology', 2008)
        ]

        for term, year_discovered in future_indicators:
            if term in content_lower and year_discovered > current_year:
                novelty_report['anachronism_detected'] = True
                novelty_report['future_knowledge_used'] = True
                novelty_report['novelty_assessment'] = f"Contains knowledge from {year_discovered} ({year_discovered - current_year} years in the future)"
                novelty_report['is_novel'] = False
                return novelty_report  # Return immediately on first anachronism detected

        # If no anachronisms detected, assess novelty relative to current knowledge
        existing_concepts = [c.content.lower() for c in self.capsules]
        novel_elements = []

        # Simple novelty detection - look for concepts not in existing knowledge
        key_phrases = ['relativity', 'quantum', 'field', 'unified', 'symmetry', 'invariance', 'entanglement', 'neural', 'blockchain']
        for phrase in key_phrases:
            if phrase in content_lower and not any(phrase in existing for existing in existing_concepts):
                novel_elements.append(phrase)

        if novel_elements:
            novelty_report['novelty_assessment'] = f"Introduces novel concepts: {', '.join(novel_elements)}"
        else:
            novelty_report['novelty_assessment'] = "Builds on existing knowledge"
            novelty_report['is_novel'] = False

        return novelty_report

    def enforce_temporal_consistency(self, capsule: Capsule) -> Dict[str, Any]:
        """Enforce temporal consistency - prevent anachronistic reasoning"""
        time_awareness = self.cayde_personality['time_awareness']

        temporal_report = {
            'temporally_consistent': True,
            'warnings': [],
            'corrections': [],
            'temporal_context': f"I am currently in year {time_awareness['current_year']}"
        }

        # Check for future knowledge contamination
        novelty_check = self.check_temporal_novelty(capsule)

        if novelty_check['anachronism_detected']:
            temporal_report['temporally_consistent'] = False
            temporal_report['warnings'].append(
                f"Warning: Anachronism detected: {novelty_check['novelty_assessment']}"
            )
            temporal_report['corrections'].append(
                f"Future knowledge from {novelty_check['novelty_assessment']} is inaccessible in {time_awareness['current_year']}"
            )

        # Add temporal context to capsule content if novel
        if novelty_check['is_novel'] and temporal_report['temporally_consistent']:
            temporal_report['corrections'].append(
                f"This idea is novel relative to year {time_awareness['current_year']}"
            )

        return temporal_report

    def perform_mathematical_operation(self, operation: str, operands: List[Any]) -> Dict[str, Any]:
        """Hard-coded mathematical operations - exact arithmetic machinery"""
        result = {
            'operation': operation,
            'operands': operands,
            'result': None,
            'valid': True,
            'error': None,
            'steps': [],
            'arithmetic_type': 'integer'  # Default to integer arithmetic
        }

        try:
            # Determine arithmetic type based on operands
            if any(isinstance(op, float) for op in operands):
                result['arithmetic_type'] = 'float_with_error'
            elif any('/' in str(op) or isinstance(op, Rational) for op in operands):
                result['arithmetic_type'] = 'rational'
            
            if operation == 'add':
                if result['arithmetic_type'] == 'rational':
                    # Convert operands to rational if needed
                    rats = [Rational(int(op.numerator), int(op.denominator)) if isinstance(op, Rational) 
                           else Rational(int(op.split('/')[0]), int(op.split('/')[1])) if '/' in str(op)
                           else Rational(int(op)) for op in operands]
                    result['result'] = rats[0]
                    for r in rats[1:]:
                        result['result'] = result['result'].add(r)
                    result['steps'].append(f"Sum: {result['result']}")
                    
                elif result['arithmetic_type'] == 'float_with_error':
                    floats = [FloatWithError(float(op.value), float(op.error)) if isinstance(op, FloatWithError)
                             else FloatWithError(float(op)) for op in operands]
                    result['result'] = floats[0]
                    for f in floats[1:]:
                        result['result'] = result['result'].add(f)
                    result['steps'].append(f"Sum with error: {result['result']}")
                    
                else:  # integer arithmetic
                    ints = [int(op) for op in operands]
                    result['result'] = ints[0]
                    for i in ints[1:]:
                        result['result'] = IntegerArithmetic.add(result['result'], i)
                    result['steps'].append(f"Integer sum: {result['result']}")

            elif operation == 'subtract':
                if len(operands) >= 2:
                    if result['arithmetic_type'] == 'rational':
                        rats = [Rational(int(op.numerator), int(op.denominator)) if isinstance(op, Rational) 
                               else Rational(int(op.split('/')[0]), int(op.split('/')[1])) if '/' in str(op)
                               else Rational(int(op)) for op in operands]
                        result['result'] = rats[0]
                        for r in rats[1:]:
                            neg_r = Rational(-r.numerator, r.denominator)
                            result['result'] = result['result'].add(neg_r)
                        result['steps'].append(f"Difference: {result['result']}")
                        
                    elif result['arithmetic_type'] == 'float_with_error':
                        floats = [FloatWithError(float(op.value), float(op.error)) if isinstance(op, FloatWithError)
                                 else FloatWithError(float(op)) for op in operands]
                        result['result'] = floats[0]
                        for f in floats[1:]:
                            neg_f = FloatWithError(-f.value, f.error)
                            result['result'] = result['result'].add(neg_f)
                        result['steps'].append(f"Difference with error: {result['result']}")
                        
                    else:  # integer arithmetic
                        ints = [int(op) for op in operands]
                        result['result'] = ints[0]
                        for i in ints[1:]:
                            result['result'] = IntegerArithmetic.add(result['result'], -i)
                        result['steps'].append(f"Integer difference: {result['result']}")

            elif operation == 'multiply':
                if result['arithmetic_type'] == 'rational':
                    rats = [Rational(int(op.numerator), int(op.denominator)) if isinstance(op, Rational) 
                           else Rational(int(op.split('/')[0]), int(op.split('/')[1])) if '/' in str(op)
                           else Rational(int(op)) for op in operands]
                    result['result'] = rats[0]
                    for r in rats[1:]:
                        result['result'] = result['result'].multiply(r)
                    result['steps'].append(f"Product: {result['result']}")
                    
                elif result['arithmetic_type'] == 'float_with_error':
                    floats = [FloatWithError(float(op.value), float(op.error)) if isinstance(op, FloatWithError)
                             else FloatWithError(float(op)) for op in operands]
                    result['result'] = floats[0]
                    for f in floats[1:]:
                        result['result'] = result['result'].multiply(f)
                    result['steps'].append(f"Product with error: {result['result']}")
                    
                else:  # integer arithmetic
                    ints = [int(op) for op in operands]
                    result['result'] = ints[0]
                    for i in ints[1:]:
                        result['result'] = IntegerArithmetic.multiply(result['result'], i)
                    result['steps'].append(f"Integer product: {result['result']}")

            elif operation == 'divide':
                if len(operands) >= 2:
                    if result['arithmetic_type'] == 'rational':
                        rats = [Rational(int(op.numerator), int(op.denominator)) if isinstance(op, Rational) 
                               else Rational(int(op.split('/')[0]), int(op.split('/')[1])) if '/' in str(op)
                               else Rational(int(op)) for op in operands]
                        result['result'] = rats[0]
                        for r in rats[1:]:
                            inv_r = Rational(r.denominator, r.numerator)
                            result['result'] = result['result'].multiply(inv_r)
                        result['steps'].append(f"Quotient: {result['result']}")
                        
                    elif result['arithmetic_type'] == 'float_with_error':
                        floats = [FloatWithError(float(op.value), float(op.error)) if isinstance(op, FloatWithError)
                                 else FloatWithError(float(op)) for op in operands]
                        result['result'] = floats[0]
                        for f in floats[1:]:
                            inv_f = FloatWithError(1.0/f.value, f.error/abs(f.value**2) if f.value != 0 else 0)
                            result['result'] = result['result'].multiply(inv_f)
                        result['steps'].append(f"Quotient with error: {result['result']}")
                        
                    else:  # For integers, result becomes rational
                        result['arithmetic_type'] = 'rational'
                        rats = [Rational(int(op)) for op in operands]
                        result['result'] = rats[0]
                        for r in rats[1:]:
                            inv_r = Rational(r.denominator, r.numerator)
                            result['result'] = result['result'].multiply(inv_r)
                        result['steps'].append(f"Integer division (rational result): {result['result']}")

            elif operation == 'power':
                if len(operands) >= 2:
                    base, exp = operands[0], operands[1]
                    
                    if result['arithmetic_type'] == 'rational':
                        base_r = Rational(int(base.numerator), int(base.denominator)) if isinstance(base, Rational) \
                                else Rational(int(base.split('/')[0]), int(base.split('/')[1])) if '/' in str(base) \
                                else Rational(int(base))
                        result['result'] = Rational(1, 1)
                        for _ in range(int(exp)):
                            result['result'] = result['result'].multiply(base_r)
                        result['steps'].append(f"Power: {result['result']}")
                        
                    elif result['arithmetic_type'] == 'float_with_error':
                        base_f = FloatWithError(float(base.value), float(base.error)) if isinstance(base, FloatWithError) \
                                else FloatWithError(float(base))
                        result['result'] = FloatWithError(1.0, 0.0)
                        for _ in range(int(exp)):
                            result['result'] = result['result'].multiply(base_f)
                        result['steps'].append(f"Power with error: {result['result']}")
                        
                    else:  # integer arithmetic
                        base_i = int(base)
                        result['result'] = 1
                        for _ in range(int(exp)):
                            result['result'] = IntegerArithmetic.multiply(result['result'], base_i)
                        result['steps'].append(f"Integer power: {result['result']}")

            elif operation == 'sqrt':
                if len(operands) >= 1:
                    # For now, use Python's math.sqrt but track it as float with error
                    import math
                    value = float(operands[0])
                    sqrt_val = math.sqrt(value)
                    # Estimate error using derivative: d(sqrt(x))/dx = 1/(2*sqrt(x))
                    error = abs(0.1 / (2 * sqrt_val)) if sqrt_val != 0 else 0  # Assume 0.1 input error
                    result['result'] = FloatWithError(sqrt_val, error)
                    result['arithmetic_type'] = 'float_with_error'
                    result['steps'].append(f"Square root with error: {result['result']}")

            # ... existing code for other operations ...
                    result['result'] = math.sqrt(operands[0])
                    result['steps'].append(f"√{operands[0]} = {result['result']}")

            elif operation == 'derivative':
                # Basic polynomial derivative
                if len(operands) == 1 and isinstance(operands[0], str):
                    result['result'] = self._compute_derivative(operands[0])
                    result['steps'].append(f"d/dx({operands[0]}) = {result['result']}")

            elif operation == 'integral':
                # Basic polynomial integral
                if len(operands) == 1 and isinstance(operands[0], str):
                    result['result'] = self._compute_integral(operands[0])
                    result['steps'].append(f"∫({operands[0]})dx = {result['result']}")

            elif operation == 'solve_quadratic':
                # ax² + bx + c = 0
                if len(operands) == 3:
                    a, b, c = operands
                    if a != 0:
                        discriminant = b**2 - 4*a*c
                        if discriminant >= 0:
                            import math
                            root1 = (-b + math.sqrt(discriminant)) / (2*a)
                            root2 = (-b - math.sqrt(discriminant)) / (2*a)
                            result['result'] = [root1, root2]
                            result['steps'].append(f"Roots of {a}x² + {b}x + {c} = 0: {root1}, {root2}")

            elif operation == 'matrix_multiply':
                if len(operands) == 2 and all(isinstance(op, list) for op in operands):
                    result['result'] = self._matrix_multiply(operands[0], operands[1])
                    result['steps'].append("Matrix multiplication completed")

            elif operation == 'vector_dot_product':
                if len(operands) == 2 and all(isinstance(op, list) for op in operands):
                    result['result'] = sum(a*b for a, b in zip(operands[0], operands[1]))
                    result['steps'].append(f"Dot product: {result['result']}")

            elif operation == 'cross_product':
                if len(operands) == 2 and all(len(op) == 3 for op in operands):
                    a, b = operands
                    result['result'] = [
                        a[1]*b[2] - a[2]*b[1],
                        a[2]*b[0] - a[0]*b[2],
                        a[0]*b[1] - a[1]*b[0]
                    ]
                    result['steps'].append(f"Cross product computed: {result['result']}")

            elif operation == 'simplify':
                if len(operands) == 1 and isinstance(operands[0], AlgebraicExpression):
                    result['result'] = AlgebraicManipulator.simplify(operands[0])
                    result['arithmetic_type'] = 'algebraic'
                    result['steps'].append(f"Simplified: {result['result']}")

            elif operation == 'expand':
                if len(operands) == 1 and isinstance(operands[0], AlgebraicExpression):
                    result['result'] = AlgebraicManipulator.expand(operands[0])
                    result['arithmetic_type'] = 'algebraic'
                    result['steps'].append(f"Expanded: {result['result']}")

            elif operation == 'factor':
                if len(operands) == 1 and isinstance(operands[0], AlgebraicExpression):
                    result['result'] = AlgebraicManipulator.factor(operands[0])
                    result['arithmetic_type'] = 'algebraic'
                    result['steps'].append(f"Factored: {result['result']}")

            elif operation == 'substitute':
                if len(operands) == 3 and isinstance(operands[0], AlgebraicExpression) and isinstance(operands[2], AlgebraicExpression):
                    expr, var_name, replacement = operands
                    result['result'] = AlgebraicManipulator.substitute(expr, var_name, replacement)
                    result['arithmetic_type'] = 'algebraic'
                    result['steps'].append(f"Substituted {var_name} with {replacement}: {result['result']}")

            elif operation == 'check_equality':
                if len(operands) == 2 and all(isinstance(op, AlgebraicExpression) for op in operands):
                    expr1, expr2 = operands
                    result['result'] = AlgebraicManipulator.check_equality(expr1, expr2)
                    result['arithmetic_type'] = 'algebraic'
                    result['steps'].append(f"Equality check: {expr1} {'=' if result['result'] else '≠'} {expr2}")

            elif operation == 'differentiate':
                if len(operands) >= 1 and isinstance(operands[0], AlgebraicExpression):
                    var_name = operands[1] if len(operands) > 1 else 'x'
                    result['result'] = CalculusEngine.differentiate(operands[0], var_name)
                    result['arithmetic_type'] = 'calculus'
                    result['steps'].append(f"d/d{var_name}({operands[0]}) = {result['result']}")

            elif operation == 'integrate':
                if len(operands) >= 1 and isinstance(operands[0], AlgebraicExpression):
                    var_name = operands[1] if len(operands) > 1 else 'x'
                    result['result'] = CalculusEngine.integrate(operands[0], var_name)
                    result['arithmetic_type'] = 'calculus'
                    result['steps'].append(f"∫({operands[0]})d{var_name} = {result['result']} + C")

            elif operation == 'numerical_integrate':
                if len(operands) >= 3:
                    func_expr, a, b = operands[:3]
                    n = operands[3] if len(operands) > 3 else 100
                    result['result'] = CalculusEngine.numerical_integrate(str(func_expr), float(a), float(b), int(n))
                    result['arithmetic_type'] = 'numerical_calculus'
                    result['steps'].append(f"∫_{a}^{b} {func_expr} dx ≈ {result['result']}")

            elif operation == 'vector_add':
                if len(operands) == 2 and all(isinstance(op, Vector) for op in operands):
                    result['result'] = LinearAlgebraEngine.vector_add(operands[0], operands[1])
                    result['arithmetic_type'] = 'linear_algebra'
                    result['steps'].append(f"{operands[0]} + {operands[1]} = {result['result']}")

            elif operation == 'vector_scalar_multiply':
                if len(operands) == 2 and isinstance(operands[0], Vector):
                    result['result'] = LinearAlgebraEngine.vector_scalar_multiply(operands[0], float(operands[1]))
                    result['arithmetic_type'] = 'linear_algebra'
                    result['steps'].append(f"{operands[1]} * {operands[0]} = {result['result']}")

            elif operation == 'vector_dot_product':
                if len(operands) == 2 and all(isinstance(op, Vector) for op in operands):
                    result['result'] = LinearAlgebraEngine.vector_dot_product(operands[0], operands[1])
                    result['arithmetic_type'] = 'linear_algebra'
                    result['steps'].append(f"{operands[0]} · {operands[1]} = {result['result']}")

            elif operation == 'matrix_add':
                if len(operands) == 2 and all(isinstance(op, Matrix) for op in operands):
                    result['result'] = LinearAlgebraEngine.matrix_add(operands[0], operands[1])
                    result['arithmetic_type'] = 'linear_algebra'
                    result['steps'].append("Matrix addition completed")

            elif operation == 'matrix_multiply':
                if len(operands) == 2 and all(isinstance(op, Matrix) for op in operands):
                    result['result'] = LinearAlgebraEngine.matrix_multiply(operands[0], operands[1])
                    result['arithmetic_type'] = 'linear_algebra'
                    result['steps'].append("Matrix multiplication completed")

            elif operation == 'matrix_transpose':
                if len(operands) == 1 and isinstance(operands[0], Matrix):
                    result['result'] = LinearAlgebraEngine.matrix_transpose(operands[0])
                    result['arithmetic_type'] = 'linear_algebra'
                    result['steps'].append("Matrix transpose completed")

            elif operation == 'matrix_determinant':
                if len(operands) == 1 and isinstance(operands[0], Matrix):
                    result['result'] = LinearAlgebraEngine.matrix_determinant(operands[0])
                    result['arithmetic_type'] = 'linear_algebra'
                    result['steps'].append(f"Determinant = {result['result']}")

            elif operation == 'matrix_vector_multiply':
                if len(operands) == 2 and isinstance(operands[0], Matrix) and isinstance(operands[1], Vector):
                    result['result'] = LinearAlgebraEngine.matrix_vector_multiply(operands[0], operands[1])
                    result['arithmetic_type'] = 'linear_algebra'
                    result['steps'].append("Matrix-vector multiplication completed")

            elif operation == 'tensor_contract':
                if len(operands) >= 3 and isinstance(operands[0], Tensor) and isinstance(operands[1], Tensor):
                    t1, t2 = operands[0], operands[1]
                    contractions = operands[2] if len(operands) > 2 else []
                    result['result'] = LinearAlgebraEngine.tensor_contract(t1, t2, contractions)
                    result['arithmetic_type'] = 'tensor_algebra'
                    result['steps'].append(f"Tensor contraction: {result['result']}")

            # ===== DIMENSIONAL ANALYSIS OPERATIONS =====
            
            elif operation == 'check_dimensional_consistency':
                """Check if an equation is dimensionally consistent"""
                if len(operands) == 2 and all(isinstance(op, list) for op in operands):
                    left_dims, right_dims = operands
                    consistency_check = DimensionalAnalysis.check_equation_consistency(left_dims, right_dims)
                    result['result'] = consistency_check
                    result['arithmetic_type'] = 'dimensional_analysis'
                    result['steps'].append(f"Dimensional consistency check: {'PASS' if consistency_check['consistent'] else 'FAIL'}")
                    if not consistency_check['consistent']:
                        result['valid'] = False
                        result['error'] = consistency_check['issues'][0]['message'] if consistency_check['issues'] else "Dimensional inconsistency"

            elif operation == 'validate_physical_meaningfulness':
                """Check if dimensions make physical sense"""
                if len(operands) == 1 and isinstance(operands[0], dict):
                    validation = DimensionalAnalysis.validate_physical_meaningfulness(operands[0])
                    result['result'] = validation
                    result['arithmetic_type'] = 'dimensional_analysis'
                    result['steps'].append(f"Physical meaningfulness check: {'PASS' if validation['meaningful'] else 'FAIL'}")
                    if not validation['meaningful'] and validation['issues']:
                        result['valid'] = False
                        result['error'] = validation['issues'][0]['message']

            elif operation == 'combine_dimensions':
                """Combine dimensions for multiplication/division"""
                if len(operands) >= 2 and all(isinstance(op, dict) for op in operands):
                    combined = operands[0]
                    for dim in operands[1:]:
                        combined = DimensionalAnalysis.multiply_dimensions(combined, dim)
                    result['result'] = combined
                    result['arithmetic_type'] = 'dimensional_analysis'
                    result['steps'].append(f"Combined dimensions: {DimensionalAnalysis.format_dimensions(combined)}")

            elif operation == 'format_dimensions':
                """Format dimensions as readable string"""
                if len(operands) == 1 and isinstance(operands[0], dict):
                    formatted = DimensionalAnalysis.format_dimensions(operands[0])
                    result['result'] = formatted
                    result['arithmetic_type'] = 'dimensional_analysis'
                    result['steps'].append(f"Formatted dimensions: {formatted}")

            else:
                result['valid'] = False
                result['error'] = f"Unknown operation: {operation}"

        except Exception as e:
            result['valid'] = False
            result['error'] = str(e)

        return result

    def _compute_derivative(self, expression: str) -> str:
        """Compute derivative of simple polynomial expressions"""
        # Very basic implementation for demonstration
        if expression == 'x^2':
            return '2x'
        elif expression == 'x^3':
            return '3x^2'
        elif expression == 'sin(x)':
            return 'cos(x)'
        elif expression == 'cos(x)':
            return '-sin(x)'
        elif expression == 'e^x':
            return 'e^x'
        else:
            return f"d/dx({expression})"  # Placeholder

    def _compute_integral(self, expression: str) -> str:
        """Compute integral of simple expressions"""
        # Very basic implementation for demonstration
        if expression == 'x':
            return 'x^2/2 + C'
        elif expression == 'x^2':
            return 'x^3/3 + C'
        elif expression == '1/x':
            return 'ln|x| + C'
        elif expression == 'cos(x)':
            return 'sin(x) + C'
        elif expression == 'sin(x)':
            return '-cos(x) + C'
        else:
            return f"∫{expression}dx + C"  # Placeholder

    def _matrix_multiply(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication"""
        if not A or not B or len(A[0]) != len(B):
            raise ValueError("Invalid matrix dimensions")

        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def validate_mathematical_expression(self, expression: str) -> Dict[str, Any]:
        """Enhanced mathematical validation that supports Cayde's reasoning without dictating beliefs"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'structure': None,
            'contradiction_events': [],  # New: track mathematical contradictions
            'uncertainty_increase': 0.0,  # New: how much uncertainty this adds
            'capsule_split_suggestions': [],  # New: suggestions for splitting inconsistent capsules
            'validation_steps': []  # New: detailed step-by-step validation
        }

        # Parse mathematical structure more deeply
        math_structure = self._parse_mathematical_structure(expression)
        validation['structure'] = math_structure

        # Enhanced validation based on structure type
        if math_structure['type'] == 'equation':
            equation_validation = self._validate_equation_structure(expression, math_structure)
            validation['validation_steps'].extend(equation_validation['steps'])
            
            if not equation_validation['consistent']:
                validation['valid'] = False
                validation['errors'].extend(equation_validation['errors'])
                validation['contradiction_events'].append({
                    'type': 'equation_inconsistency',
                    'description': f"Equation '{expression}' contains dimensional or logical inconsistencies",
                    'severity': equation_validation['severity'],
                    'suggested_resolution': equation_validation['resolution']
                })
                validation['uncertainty_increase'] = equation_validation['uncertainty_penalty']

        elif math_structure['type'] == 'derivation':
            derivation_validation = self._validate_derivation_steps(expression, math_structure)
            validation['validation_steps'].extend(derivation_validation['steps'])
            
            if not derivation_validation['valid']:
                validation['valid'] = False
                validation['errors'].extend(derivation_validation['errors'])
                validation['contradiction_events'].extend(derivation_validation['contradictions'])
                validation['uncertainty_increase'] = derivation_validation['uncertainty_penalty']
                
                # Suggest capsule splitting for derivation failures
                if derivation_validation['severity'] == 'high':
                    validation['capsule_split_suggestions'].append({
                        'reason': 'mathematical_derivation_failure',
                        'description': 'Derivation contains invalid mathematical steps - consider splitting into valid and invalid components',
                        'split_points': derivation_validation['invalid_steps']
                    })

        elif math_structure['type'] == 'definition':
            definition_validation = self._validate_mathematical_definition(expression, math_structure)
            validation['validation_steps'].extend(definition_validation['steps'])
            
            if not definition_validation['valid']:
                validation['valid'] = False
                validation['errors'].extend(definition_validation['errors'])
                validation['contradiction_events'].append({
                    'type': 'definition_inconsistency',
                    'description': f"Mathematical definition '{expression}' is inconsistent",
                    'severity': 'medium'
                })
                validation['uncertainty_increase'] = 0.2

        # Basic structural checks (keep existing functionality)
        open_count = expression.count('(')
        close_count = expression.count(')')
        if open_count != close_count:
            validation['valid'] = False
            validation['errors'].append("Unbalanced parentheses")
            validation['uncertainty_increase'] = max(validation['uncertainty_increase'], 0.1)

        if '/0' in expression or '/ 0' in expression:
            validation['valid'] = False
            validation['errors'].append("Division by zero")
            validation['contradiction_events'].append({
                'type': 'division_by_zero',
                'description': 'Division by zero creates mathematical singularity',
                'severity': 'high'
            })
            validation['uncertainty_increase'] = max(validation['uncertainty_increase'], 0.5)

        if '0^0' in expression:
            validation['warnings'].append("0^0 is undefined")
            validation['uncertainty_increase'] = max(validation['uncertainty_increase'], 0.3)

        return validation

    def _parse_mathematical_structure(self, expression: str) -> Dict[str, Any]:
        """Parse the mathematical structure of an expression"""
        structure = {
            'type': 'expression',
            'components': [],
            'operators': [],
            'variables': set(),
            'constants': set()
        }

        # Detect equation structure
        if '=' in expression:
            sides = expression.split('=')
            if len(sides) == 2:
                structure['type'] = 'equation'
                structure['left_side'] = sides[0].strip()
                structure['right_side'] = sides[1].strip()
                
                # Check for derivation indicators
                if '→' in expression or '⇒' in expression or any(step in expression.lower() for step in ['step', 'therefore', 'hence', 'thus']):
                    structure['type'] = 'derivation'
                    structure['steps'] = self._extract_derivation_steps(expression)

        # Detect definition patterns
        elif any(pattern in expression.lower() for pattern in ['defined as', 'denotes', 'represents', 'is defined']):
            structure['type'] = 'definition'
            structure['defined_term'] = self._extract_defined_term(expression)

        # Extract variables and constants
        import re
        variables = re.findall(r'\b[a-zA-Z]+\b', expression)
        structure['variables'] = set(variables)
        
        # Simple constant detection (numbers)
        constants = re.findall(r'\b\d+\.?\d*\b', expression)
        structure['constants'] = set(constants)

        return structure

    def _validate_equation_structure(self, expression: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate equation structure using dimensional analysis"""
        validation = {
            'consistent': True,
            'errors': [],
            'steps': [],
            'severity': 'low',
            'uncertainty_penalty': 0.0,
            'resolution': None
        }

        left_side = structure.get('left_side', '')
        right_side = structure.get('right_side', '')

        # Try dimensional analysis if we can extract dimensions
        left_dims = self._extract_dimensions_from_expression(left_side)
        right_dims = self._extract_dimensions_from_expression(right_side)

        if left_dims and right_dims:
            consistency_check = DimensionalAnalysis.check_equation_consistency([left_dims], [right_dims])
            validation['steps'].append(f"Dimensional analysis: {DimensionalAnalysis.format_dimensions(left_dims)} = {DimensionalAnalysis.format_dimensions(right_dims)}")
            
            if not consistency_check['consistent']:
                validation['consistent'] = False
                validation['errors'].append(f"Dimensional inconsistency: {consistency_check['issues'][0]['message']}")
                validation['severity'] = 'high'
                validation['uncertainty_penalty'] = 0.4
                validation['resolution'] = "Equation violates dimensional consistency - check units"
            else:
                validation['steps'].append("✓ Dimensional consistency verified")

        # Check for obvious mathematical impossibilities
        if self._contains_mathematical_impossibility(expression):
            validation['consistent'] = False
            validation['errors'].append("Expression contains mathematically impossible operations")
            validation['severity'] = 'high'
            validation['uncertainty_penalty'] = 0.6

        return validation

    def _validate_derivation_steps(self, expression: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mathematical derivation steps"""
        validation = {
            'valid': True,
            'errors': [],
            'steps': [],
            'contradictions': [],
            'severity': 'low',
            'uncertainty_penalty': 0.0,
            'invalid_steps': []
        }

        steps = structure.get('steps', [])
        
        for i, step in enumerate(steps):
            step_validation = self._validate_single_derivation_step(step, i, steps)
            validation['steps'].append(f"Step {i+1}: {step_validation['description']}")
            
            if not step_validation['valid']:
                validation['valid'] = False
                validation['errors'].append(f"Step {i+1} invalid: {step_validation['error']}")
                validation['contradictions'].append({
                    'type': 'invalid_derivation_step',
                    'step_number': i+1,
                    'description': step_validation['error'],
                    'severity': step_validation['severity']
                })
                validation['invalid_steps'].append(i+1)
                validation['severity'] = max(validation['severity'], step_validation['severity'], key=lambda x: {'low': 0, 'medium': 1, 'high': 2}[x])
                validation['uncertainty_penalty'] = max(validation['uncertainty_penalty'], step_validation['uncertainty_penalty'])

        return validation

    def _validate_single_derivation_step(self, step: str, step_number: int, all_steps: List[str]) -> Dict[str, Any]:
        """Validate a single step in a mathematical derivation"""
        validation = {
            'valid': True,
            'error': None,
            'description': f"Step {step_number + 1} appears valid",
            'severity': 'low',
            'uncertainty_penalty': 0.0
        }

        # Check for dimensional consistency within the step
        if '=' in step:
            parts = step.split('=')
            if len(parts) == 2:
                left_dims = self._extract_dimensions_from_expression(parts[0])
                right_dims = self._extract_dimensions_from_expression(parts[1])
                
                if left_dims and right_dims:
                    consistency = DimensionalAnalysis.check_equation_consistency([left_dims], [right_dims])
                    if not consistency['consistent']:
                        validation['valid'] = False
                        validation['error'] = f"Dimensional inconsistency in step: {consistency['issues'][0]['message']}"
                        validation['severity'] = 'high'
                        validation['uncertainty_penalty'] = 0.3

        # Check for mathematical impossibilities
        if self._contains_mathematical_impossibility(step):
            validation['valid'] = False
            validation['error'] = "Step contains mathematically impossible operations"
            validation['severity'] = 'high'
            validation['uncertainty_penalty'] = 0.5

        return validation

    def _validate_mathematical_definition(self, expression: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mathematical definitions"""
        validation = {
            'valid': True,
            'errors': [],
            'steps': []
        }

        # Basic validation - definitions should be consistent with existing mathematical knowledge
        defined_term = structure.get('defined_term', '')
        
        # Check if the definition conflicts with established mathematical conventions
        if defined_term:
            conflicts = self._check_definition_conflicts(defined_term, expression)
            if conflicts:
                validation['valid'] = False
                validation['errors'].extend(conflicts)

        validation['steps'].append(f"Definition validation for '{defined_term}': {'valid' if validation['valid'] else 'invalid'}")
        
        return validation

    def _extract_dimensions_from_expression(self, expression: str) -> Optional[Dict[str, int]]:
        """Extract dimensional information from a mathematical expression"""
        # Simple dimensional extraction - in practice this would be more sophisticated
        expr_lower = expression.lower().strip()
        
        # Common physical quantities and their dimensions
        dimension_map = {
            # Basic dimensions
            'length': {'L': 1},
            'mass': {'M': 1},
            'time': {'T': 1},
            'charge': {'Q': 1},
            'temperature': {'Θ': 1},
            
            # Derived quantities
            'velocity': {'L': 1, 'T': -1},
            'acceleration': {'L': 1, 'T': -2},
            'force': {'M': 1, 'L': 1, 'T': -2},
            'energy': {'M': 1, 'L': 2, 'T': -2},
            'power': {'M': 1, 'L': 2, 'T': -3},
            'pressure': {'M': 1, 'L': -1, 'T': -2},
            
            # Mathematical operations
            'area': {'L': 2},
            'volume': {'L': 3},
        }
        
        for term, dims in dimension_map.items():
            if term in expr_lower or f'{term}s' in expr_lower:  # Handle plural
                return dims
                
        # If no specific term found, try to infer from mathematical structure
        if 'v' in expr_lower and ('t' in expr_lower or 'time' in expr_lower):
            return {'L': 1, 'T': -1}  # Likely velocity
        elif 'a' in expr_lower and ('t' in expr_lower or 'time' in expr_lower):
            return {'L': 1, 'T': -2}  # Likely acceleration
        elif 'f' in expr_lower and ('m' in expr_lower or 'mass' in expr_lower):
            return {'M': 1, 'L': 1, 'T': -2}  # Likely force
            
        return None  # Cannot determine dimensions

    def _contains_mathematical_impossibility(self, expression: str) -> bool:
        """Check for mathematically impossible operations"""
        impossibilities = [
            '∞/∞', '0/0', '∞-∞', '0*∞', '∞^0',
            'log(0)', 'log(-1)',
        ]
        
        # Check for sqrt of negative numbers
        if 'sqrt' in expression and '-1' in expression:
            impossibilities.append('sqrt(-1)')
        
        expr_clean = expression.replace(' ', '')
        return any(imp in expr_clean for imp in impossibilities)

    def _extract_derivation_steps(self, expression: str) -> List[str]:
        """Extract individual steps from a mathematical derivation"""
        # Simple step extraction - split on common step indicators
        step_indicators = ['→', '⇒', 'therefore', 'hence', 'thus', 'step', '\n']
        
        steps = [expression]  # Default: whole expression as one step
        
        for indicator in step_indicators:
            if indicator in expression:
                if indicator == '\n':
                    steps = [line.strip() for line in expression.split('\n') if line.strip()]
                else:
                    steps = [part.strip() for part in expression.split(indicator) if part.strip()]
                break
                
        return steps

    def _extract_defined_term(self, expression: str) -> str:
        """Extract the term being defined"""
        # Simple extraction - look for patterns like "X is defined as"
        import re
        match = re.search(r'(\w+)\s+is defined as', expression, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""

    def _check_definition_conflicts(self, term: str, definition: str) -> List[str]:
        """Check if a mathematical definition conflicts with established conventions"""
        conflicts = []
        
        # Example conflicts (would be expanded with actual mathematical knowledge)
        if term.lower() == 'pi' and '3' in definition and '14159' not in definition:
            conflicts.append("π is conventionally defined as approximately 3.14159..., not exactly 3")
            
        return conflicts

    def create_contradiction_event(self, capsule: Capsule, contradiction: Dict[str, Any]) -> None:
        """Create a contradiction event that increases uncertainty and may trigger capsule evolution"""
        # Record the contradiction in the capsule
        capsule.intellectual_conflicts.append({
            'type': 'mathematical_contradiction',
            'description': contradiction['description'],
            'severity': contradiction['severity'],
            'timestamp': int(time.time() * 1000),
            'resolution': contradiction.get('suggested_resolution')
        })
        
        # Increase uncertainty based on contradiction severity
        uncertainty_increase = {
            'low': 0.1,
            'medium': 0.25,
            'high': 0.4
        }.get(contradiction['severity'], 0.2)
        
        capsule.certainty = max(0.1, capsule.certainty - uncertainty_increase)
        capsule.pose['certainty'] = max(0.1, capsule.pose['certainty'] - uncertainty_increase)
        
        print(f"⚡ Mathematical contradiction detected: {contradiction['description']}")
        print(f"   📉 Capsule certainty reduced by {uncertainty_increase:.2f} (now {capsule.certainty:.2f})")

    def attempt_capsule_split(self, capsule: Capsule, split_suggestion: Dict[str, Any]) -> Optional[List[Capsule]]:
        """Attempt to split a capsule when mathematical contradictions are detected"""
        if capsule.locked:
            print("🔒 Capsule is locked - cannot split")
            return None
            
        if split_suggestion['reason'] == 'mathematical_derivation_failure':
            return self._split_derivation_capsule(capsule, split_suggestion)
            
        return None

    def _split_derivation_capsule(self, capsule: Capsule, split_suggestion: Dict[str, Any]) -> List[Capsule]:
        """Split a derivation capsule into valid and invalid components"""
        invalid_steps = split_suggestion.get('invalid_steps', [])
        
        # Create valid capsule with only valid steps
        valid_content = self._extract_valid_derivation_steps(capsule.content, invalid_steps)
        valid_capsule = Capsule(
            content=valid_content,
            perspective=capsule.perspective,
            character=capsule.character,
            persona=capsule.persona,
            kind=capsule.kind,
            certainty=capsule.certainty * 0.8,  # Slightly reduced certainty
            parent=capsule
        )
        
        # Create invalid capsule with problematic steps
        invalid_content = self._extract_invalid_derivation_steps(capsule.content, invalid_steps)
        invalid_capsule = Capsule(
            content=invalid_content,
            perspective=capsule.perspective,
            character=capsule.character,
            persona=capsule.persona,
            kind=CapsuleKind.CONCEPT,  # Mark as concept, not theory
            certainty=capsule.certainty * 0.3,  # Much lower certainty
            parent=capsule,
            success_status="failed"  # Mark as failed
        )
        
        print(f"🔀 Capsule split due to mathematical contradictions:")
        print(f"   ✅ Valid component: {valid_content[:50]}...")
        print(f"   ❌ Invalid component: {invalid_content[:50]}...")
        
        return [valid_capsule, invalid_capsule]

    def _extract_valid_derivation_steps(self, content: str, invalid_steps: List[int]) -> str:
        """Extract valid steps from a derivation"""
        steps = self._extract_derivation_steps(content)
        valid_steps = [step for i, step in enumerate(steps) if i+1 not in invalid_steps]
        return " → ".join(valid_steps)

    def _extract_invalid_derivation_steps(self, content: str, invalid_steps: List[int]) -> str:
        """Extract invalid steps from a derivation"""
        steps = self._extract_derivation_steps(content)
        invalid_steps_content = [step for i, step in enumerate(steps) if i+1 in invalid_steps]
        return " → ".join(invalid_steps_content)

    def apply_mathematical_philosophy(self, mathematical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Cayde's mathematical philosophy to interpret results - shapes reasoning style"""
        personality = self.cayde_personality
        philosophically_interpreted = mathematical_result.copy()

        # Apply platonism - belief that math is discovered
        if personality['mathematical_platonism']['level'] > 0.7:
            philosophically_interpreted['philosophical_note'] = "This mathematical truth was discovered, not invented"
            philosophically_interpreted['confidence_multiplier'] = 1.2  # Higher confidence in "eternal" truths

        # Apply geometric realism - geometry is about physical space
        if personality['geometric_realism']['level'] > 0.8:
            if any(term in str(mathematical_result.get('operation', '')) for term in ['geometry', 'space', 'manifold']):
                philosophically_interpreted['philosophical_note'] = "This geometry describes physical reality itself"
                philosophically_interpreted['physical_interpretation'] = True

        # Apply mathematical skepticism - question necessity (overrides platonism if stronger)
        if personality['mathematical_skepticism']['level'] > 0.5:
            philosophically_interpreted['skeptical_note'] = "Is this mathematical structure physically necessary, or merely useful?"
            philosophically_interpreted['confidence_multiplier'] = 0.9  # Lower confidence without physical justification

        return philosophically_interpreted

    def _contains_mathematical_expressions(self, content: str) -> bool:
        """Check if content contains mathematical expressions"""
        content_lower = content.lower()

        # Mathematical operators and symbols
        math_indicators = [
            '=', '+', '-', '*', '/', '^', '√', '∫', '∂', '∑', '∏',
            'sin(', 'cos(', 'tan(', 'log(', 'ln(', 'exp(',
            'dx', 'dy', 'dz', 'dt',
            'matrix', 'vector', 'tensor',
            'derivative', 'integral', 'differential'
        ]

        # Check for numbers and mathematical patterns
        import re
        if re.search(r'\d', content):  # Contains digits
            return True

        for indicator in math_indicators:
            if indicator in content_lower:
                return True

        # Check for equations (contains = with math-like context)
        if '=' in content and any(term in content_lower for term in ['equation', 'formula', 'x', 'y', 'z']):
            return True

        return False

    def parse_symbolic_expression(self, expression: str) -> Dict[str, Any]:
        """Hard-coded symbolic parsing - parse mathematical and logical expressions"""
        result = {
            'expression': expression,
            'parsed': True,
            'structure': None,
            'symbols': [],
            'operators': [],
            'errors': []
        }

        try:
            # Basic symbolic parsing
            import re

            # Extract symbols (variables, constants)
            symbols = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
            result['symbols'] = list(set(symbols))

            # Extract operators
            operators = re.findall(r'[+\-*/=<>!&|%^~]', expression)
            result['operators'] = list(set(operators))

            # Determine structure type
            if '=' in expression and ('+' in expression or '-' in expression or '*' in expression or '/' in expression):
                result['structure'] = 'equation'
            elif any(op in expression for op in ['∀', '∃', '→', '↔', '¬']):
                result['structure'] = 'logical'
            elif any(func in expression.lower() for func in ['sin', 'cos', 'tan', 'log', 'exp']):
                result['structure'] = 'functional'
            else:
                result['structure'] = 'expression'

        except Exception as e:
            result['parsed'] = False
            result['errors'].append(str(e))

        return result

    def check_grammar_structure(self, text: str) -> Dict[str, Any]:
        """Hard-coded grammar checking - validate sentence structure and syntax"""
        validation = {
            'text': text,
            'valid': True,
            'issues': [],
            'structure_score': 1.0,
            'readability_score': 1.0
        }

        # Basic grammar checks
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check sentence structure
            words = sentence.split()
            if len(words) < 3:
                validation['issues'].append("Sentence too short")
                validation['structure_score'] *= 0.9

            # Check capitalization
            if sentence and not sentence[0].isupper():
                validation['issues'].append("Sentence doesn't start with capital letter")
                validation['valid'] = False

            # Check punctuation
            if not sentence.endswith(('!', '?', '.')) and len(sentences) > 1:
                validation['issues'].append("Missing end punctuation")
                validation['valid'] = False

        # Readability checks
        avg_words_per_sentence = len(text.split()) / max(1, len(sentences))
        if avg_words_per_sentence > 25:
            validation['issues'].append("Sentences too long")
            validation['readability_score'] *= 0.8
        elif avg_words_per_sentence < 5:
            validation['issues'].append("Sentences too short")
            validation['readability_score'] *= 0.9

        return validation

    def extract_semantic_relationships(self, text: str) -> Dict[str, Any]:
        """Hard-coded semantic parsing - extract relationships between concepts"""
        relationships = {
            'text': text,
            'relationships': [],
            'concepts': [],
            'causal_links': [],
            'definitional_links': []
        }

        # Extract concepts (nouns and noun phrases)
        import re
        concepts = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words as concepts
        relationships['concepts'] = list(set(concepts))

        # Find causal relationships
        causal_indicators = ['because', 'therefore', 'thus', 'hence', 'consequently', 'leads to', 'causes']
        for indicator in causal_indicators:
            if indicator in text.lower():
                relationships['causal_links'].append(indicator)

        # Find definitional relationships
        definitional_indicators = ['is defined as', 'means', 'refers to', 'represents', 'denotes']
        for indicator in definitional_indicators:
            if indicator in text.lower():
                relationships['definitional_links'].append(indicator)

        # Extract basic relationships
        if 'is' in text.lower():
            relationships['relationships'].append('definition/equality')
        if 'causes' in text.lower() or 'leads to' in text.lower():
            relationships['relationships'].append('causation')
        if 'similar to' in text.lower() or 'like' in text.lower():
            relationships['relationships'].append('similarity')

        return relationships

    def process_symbolic_content(self, content: str) -> Dict[str, Any]:
        """Process content containing symbolic expressions"""
        processing_result = {
            'content': content,
            'symbolic_expressions': [],
            'parsed_expressions': [],
            'semantic_analysis': None,
            'language_patterns_learned': 0
        }

        # Extract potential symbolic expressions
        import re
        # Look for mathematical expressions
        math_patterns = [
            r'\b\d+[\d\s]*[+\-*/=<>!&|%^~][\d\s]*\d+\b',  # Basic math
            r'[a-zA-Z_][a-zA-Z0-9_]*\s*[+\-*/=<>!&|%^~]\s*[a-zA-Z0-9_]+',  # Variable expressions
            r'\b(sin|cos|tan|log|exp|sqrt)\s*\([^)]+\)',  # Functions
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, content)
            processing_result['symbolic_expressions'].extend(matches)

        # Parse each symbolic expression
        for expr in processing_result['symbolic_expressions']:
            parsed = self.parse_symbolic_expression(expr)
            processing_result['parsed_expressions'].append(parsed)

        # Perform semantic analysis
        processing_result['semantic_analysis'] = self.extract_semantic_relationships(content)

        # Learn language patterns from symbolic content
        if processing_result['symbolic_expressions']:
            self.learn_language_pattern('historical_language',
                f"Symbolic expression: {processing_result['symbolic_expressions'][0]}",
                'symbolic_processing')
            processing_result['language_patterns_learned'] += 1

        return processing_result

    def learn_language_pattern(self, pattern_type: str, example: str, context: str = ""):
        """Learn language patterns - metaphor, analogy, explanatory style, historical shifts"""
        personality = self.cayde_personality

        # Update appropriate language learning trait
        if pattern_type == 'metaphor':
            personality['metaphorical_understanding']['level'] = min(1.0,
                personality['metaphorical_understanding']['level'] + 0.05)
        elif pattern_type == 'analogy':
            personality['analogical_reasoning']['level'] = min(1.0,
                personality['analogical_reasoning']['level'] + 0.05)
        elif pattern_type == 'explanatory_style':
            personality['explanatory_style']['level'] = min(1.0,
                personality['explanatory_style']['level'] + 0.03)
        elif pattern_type == 'historical_language':
            personality['historical_language_awareness']['level'] = min(1.0,
                personality['historical_language_awareness']['level'] + 0.04)

        # Store the learned pattern
        if 'learned_language_patterns' not in personality:
            personality['learned_language_patterns'] = []

        personality['learned_language_patterns'].append({
            'type': pattern_type,
            'example': example,
            'context': context,
            'timestamp': time.time()
        })

        print(f"📚 Learned {pattern_type}: {example[:50]}...")

    def generate_analogy(self, source_concept: str, target_domain: str) -> str:
        """Generate analogy using learned patterns - shaped by personality"""
        personality = self.cayde_personality
        analogy_level = personality['analogical_reasoning']['level']

        if analogy_level < 0.3:
            return f"{source_concept} is like {target_domain} in some way."

        # Generate more sophisticated analogies based on learning level
        if analogy_level > 0.7:
            return f"Just as {source_concept} revolutionized physics, {target_domain} transforms our understanding of reality."
        elif analogy_level > 0.5:
            return f"{source_concept} relates to {target_domain} through fundamental principles of transformation."
        else:
            return f"{source_concept} can be understood by considering {target_domain}."

    def apply_explanatory_style(self, concept: str, audience: str = "general") -> str:
        """Apply learned explanatory style - shaped by personality development"""
        personality = self.cayde_personality
        style_level = personality['explanatory_style']['level']

        if style_level < 0.2:
            return f"{concept} is an important idea."

        # Apply more sophisticated explanatory styles based on development
        if audience == "scientific":
            if style_level > 0.8:
                return f"The principle of {concept} emerges from the necessity of reconciling apparently contradictory phenomena."
            elif style_level > 0.6:
                return f"{concept} represents a fundamental symmetry in nature's laws."
            else:
                return f"{concept} is a key concept in understanding physical reality."
        else:
            if style_level > 0.8:
                return f"Imagine {concept} as the hidden rhythm that orchestrates the cosmic dance."
            elif style_level > 0.6:
                return f"{concept} reveals how the universe maintains its elegant consistency."
            else:
                return f"{concept} helps us understand how things really work."

    def explain_concept_with_style(self, concept: str, domain: str = "physics") -> str:
        """Explain a concept using learned explanatory style and analogies"""
        personality = self.cayde_personality

        # Base explanation
        explanation = self.apply_explanatory_style(concept, "general")

        # Add analogy if reasoning level is high enough
        if personality['analogical_reasoning']['level'] > 0.4:
            if domain == "physics":
                analogy = self.generate_analogy(concept, "light bending through spacetime")
            elif domain == "mathematics":
                analogy = self.generate_analogy(concept, "the language of nature's patterns")
            else:
                analogy = self.generate_analogy(concept, "human understanding")

            explanation += f" {analogy}"

        # Add metaphorical elements if understanding is developed
        if personality['metaphorical_understanding']['level'] > 0.6:
            metaphors = [
                "like a hidden melody in the symphony of the cosmos",
                "as the quiet revolution that reshapes our mental landscape",
                "the bridge between what we see and what truly is"
            ]
            explanation += f" It emerges {metaphors[len(explanation) % len(metaphors)]}."

        return explanation

    def analyze_text_with_language_awareness(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis using all language processing capabilities"""
        analysis = {
            'text': text,
            'grammar_validation': self.check_grammar_structure(text),
            'semantic_relationships': self.extract_semantic_relationships(text),
            'symbolic_processing': self.process_symbolic_content(text),
            'language_maturity_assessment': {},
            'recommendations': []
        }

        # Assess language maturity based on personality development
        personality = self.cayde_personality
        maturity_score = (
            personality['metaphorical_understanding']['level'] +
            personality['analogical_reasoning']['level'] +
            personality['explanatory_style']['level'] +
            personality['historical_language_awareness']['level']
        ) / 4.0

        analysis['language_maturity_assessment'] = {
            'overall_score': maturity_score,
            'metaphorical_understanding': personality['metaphorical_understanding']['level'],
            'analogical_reasoning': personality['analogical_reasoning']['level'],
            'explanatory_style': personality['explanatory_style']['level'],
            'historical_language_awareness': personality['historical_language_awareness']['level']
        }

        # Generate recommendations
        if maturity_score < 0.3:
            analysis['recommendations'].append("Language capabilities are developing - focus on basic pattern recognition")
        elif maturity_score < 0.7:
            analysis['recommendations'].append("Good progress in language learning - continue building analogies and metaphors")
        else:
            analysis['recommendations'].append("Advanced language capabilities achieved - ready for sophisticated communication")

        if not analysis['grammar_validation']['valid']:
            analysis['recommendations'].append("Improve grammatical structure for better clarity")

        if analysis['symbolic_processing']['symbolic_expressions']:
            analysis['recommendations'].append("Strong symbolic processing capabilities detected")

        return analysis

    def argue_with_past_self(self, current_topic: str, current_position: str) -> Dict[str, Any]:
        """Generate a dialogue between current Cayde and past versions of itself"""
        personality = self.cayde_personality

        # Get past positions from personality evolution history
        past_positions = personality.get('personality_evolution', [])
        if not past_positions:
            return {'dialogue': [], 'evolution_insights': [], 'maturity_level': 0.0}

        dialogue = []
        evolution_insights = []

        # Find relevant past positions
        relevant_past = [p for p in past_positions if current_topic.lower() in p.get('topic', '').lower()]

        for past_pos in relevant_past[-3:]:  # Look at last 3 relevant past positions
            past_argument = {
                'past_self': {
                    'position': past_pos.get('position', 'unclear'),
                    'confidence': past_pos.get('confidence', 0.5),
                    'reasoning': past_pos.get('reasoning', 'limited perspective')
                },
                'current_self': {
                    'position': current_position,
                    'confidence': personality['self_critical_reflection']['level'],
                    'critique': self._generate_self_critique(past_pos, current_position)
                }
            }
            dialogue.append(past_argument)

            # Extract evolution insights
            insight = self._extract_evolution_insight(past_pos, current_position)
            if insight:
                evolution_insights.append(insight)

        # Update self-critical reflection level
        personality['self_critical_reflection']['level'] = min(1.0,
            personality['self_critical_reflection']['level'] + 0.02)

        return {
            'dialogue': dialogue,
            'evolution_insights': evolution_insights,
            'maturity_level': personality['self_critical_reflection']['level']
        }

    def _generate_self_critique(self, past_position: Dict, current_position: str) -> str:
        """Generate a critique of past thinking from current perspective"""
        past_pos = past_position.get('position', '')
        past_reasoning = past_position.get('reasoning', '')

        critiques = [
            f"My earlier view of '{past_pos}' was too simplistic. Now I see that {current_position}.",
            f"I used to think {past_reasoning}, but experience has shown me that {current_position}.",
            f"My past self was limited by {past_reasoning}. Current understanding reveals {current_position}.",
            f"Looking back, my previous position on {past_pos} lacked the depth that {current_position} provides."
        ]

        return critiques[len(past_reasoning) % len(critiques)]

    def _extract_evolution_insight(self, past_position: Dict, current_position: str) -> str:
        """Extract insights about intellectual evolution"""
        past_year = past_position.get('year', 1905)
        current_year = self.cayde_personality['time_awareness']['current_year']

        if current_year > past_year:
            return f"Evolution from {past_year} to {current_year}: Gained deeper understanding through experience."
        return "Continuous refinement of thinking through self-reflection."

    def explain_historical_success(self, concept: str, winner: str, losers: List[str]) -> Dict[str, Any]:
        """Explain why certain ideas succeeded historically over alternatives"""
        personality = self.cayde_personality

        analysis = {
            'concept': concept,
            'winning_approach': winner,
            'alternative_approaches': losers,
            'success_factors': [],
            'failure_reasons': [],
            'historical_insights': [],
            'lessons_learned': []
        }

        # Analyze success factors based on historical patterns
        if 'relativity' in concept.lower():
            analysis['success_factors'] = [
                'Unified previously separate phenomena',
                'Made novel, testable predictions',
                'Resolved existing experimental anomalies',
                'Provided deeper conceptual foundation'
            ]
            analysis['failure_reasons'] = [
                'Maintained outdated conceptual frameworks',
                'Failed to predict new phenomena',
                'Ignored experimental evidence'
            ]
        elif 'quantum' in concept.lower():
            analysis['success_factors'] = [
                'Explained microscopic phenomena unexplainable classically',
                'Provided accurate quantitative predictions',
                'Revealed fundamental limits of classical physics'
            ]
            analysis['failure_reasons'] = [
                'Attempted to force classical intuitions onto quantum domain',
                'Ignored statistical nature of microscopic processes'
            ]
        else:
            # Generic analysis
            analysis['success_factors'] = [
                'Better agreement with experimental evidence',
                'Greater explanatory power',
                'Conceptual elegance and unification',
                'Novel predictions confirmed by experiment'
            ]
            analysis['failure_reasons'] = [
                'Inadequate explanatory power',
                'Contradiction with established evidence',
                'Conceptual complexity without benefit'
            ]

        # Generate historical insights
        analysis['historical_insights'] = [
            f"The success of {winner} demonstrates that scientific progress often requires abandoning cherished intuitions.",
            f"Historical alternatives like {', '.join(losers)} failed because they maintained outdated frameworks.",
            f"True innovation requires both conceptual boldness and empirical grounding."
        ]

        # Extract lessons
        analysis['lessons_learned'] = [
            "Scientific theories succeed when they resolve anomalies and make novel predictions.",
            "Failure to abandon limiting assumptions prevents paradigm advancement.",
            "Historical awareness prevents repeating past mistakes."
        ]

        # Update historical success analysis level
        personality['historical_success_analysis']['level'] = min(1.0,
            personality['historical_success_analysis']['level'] + 0.03)

        return analysis

    def propose_historically_aware_alternatives(self, problem: str, current_approaches: List[str]) -> Dict[str, Any]:
        """Propose alternatives while being aware of historical successes and failures"""
        personality = self.cayde_personality

        alternatives = {
            'problem': problem,
            'current_approaches': current_approaches,
            'novel_alternatives': [],
            'historical_awareness': [],
            'dead_ends_avoided': [],
            'innovation_potential': 0.0
        }

        # Generate alternatives based on problem type
        if 'gravity' in problem.lower() or 'gravitation' in problem.lower():
            alternatives['novel_alternatives'] = [
                'Geometric interpretation of gravity as spacetime curvature',
                'Quantum field theory approach to gravitational interactions',
                'Emergent gravity from thermodynamic principles'
            ]
            alternatives['historical_awareness'] = [
                'Newtonian action-at-a-distance avoided as conceptually problematic',
                'Cartesian coordinate systems recognized as limiting',
                'Historical attempts at absolute space rejected'
            ]
            alternatives['dead_ends_avoided'] = [
                'Emission theories of gravity (failed historically)',
                'Absolute space frameworks (rejected by relativity)',
                'Purely mechanical explanations (insufficient for general relativity)'
            ]
        elif 'quantum' in problem.lower():
            alternatives['novel_alternatives'] = [
                'Many-worlds interpretation with decoherence',
                'Bohmian mechanics with nonlocal guidance',
                'Relational interpretation avoiding measurement problem'
            ]
            alternatives['historical_awareness'] = [
                'Classical intuitions about determinism abandoned',
                'Particle-wave duality embraced over either-or thinking',
                'Statistical interpretation accepted over deterministic hopes'
            ]
            alternatives['dead_ends_avoided'] = [
                'Hidden variable theories assuming locality (Bell inequality violations)',
                'Classical wave theories without quantization',
                'Deterministic interpretations incompatible with uncertainty principle'
            ]
        else:
            # Generic alternatives
            alternatives['novel_alternatives'] = [
                'Unified framework combining multiple perspectives',
                'Geometric interpretation of the phenomenon',
                'Emergent behavior from simpler principles'
            ]
            alternatives['historical_awareness'] = [
                'Past failed attempts inform current approach',
                'Historical dead ends avoided through pattern recognition',
                'Successful historical strategies emulated'
            ]
            alternatives['dead_ends_avoided'] = [
                'Overly complex mathematical formalisms',
                'Intuitive but incorrect analogies',
                'Ignoring experimental constraints'
            ]

        # Calculate innovation potential based on maturity
        innovation_level = personality['historically_aware_innovation']['level']
        alternatives['innovation_potential'] = min(1.0, innovation_level + 0.1)

        # Update innovation level
        personality['historically_aware_innovation']['level'] = min(1.0,
            personality['historically_aware_innovation']['level'] + 0.02)

        return alternatives

    def avoid_known_dead_ends(self, proposed_idea: str) -> Dict[str, Any]:
        """Check if a proposed idea repeats historically known failures"""
        personality = self.cayde_personality

        assessment = {
            'proposed_idea': proposed_idea,
            'dead_end_risks': [],
            'historical_precedents': [],
            'risk_level': 0.0,
            'recommendations': [],
            'safe_to_pursue': True
        }

        # Check for known failure patterns
        failure_patterns = {
            'emission theory': ['light emission', 'particle emission', 'mechanical emission'],
            'absolute space': ['fixed reference', 'absolute frame', 'privileged frame'],
            'hidden variables': ['local hidden', 'deterministic hidden', 'classical hidden'],
            'luminiferous aether': ['aether drag', 'aether wind', 'stationary aether'],
            'caloric fluid': ['heat fluid', 'imponderable fluid', 'material heat'],
            'phlogiston': ['combustion principle', 'fire element', 'negative weight']
        }

        for dead_end, indicators in failure_patterns.items():
            for indicator in indicators:
                if indicator.lower() in proposed_idea.lower():
                    assessment['dead_end_risks'].append({
                        'pattern': dead_end,
                        'indicator': indicator,
                        'historical_failure': f"{dead_end} was rejected due to empirical evidence"
                    })
                    assessment['risk_level'] += 0.3
                    assessment['safe_to_pursue'] = False

        # Generate recommendations
        if assessment['risk_level'] > 0.5:
            assessment['recommendations'].append("High risk of repeating historical failure - reconsider approach")
        elif assessment['risk_level'] > 0.2:
            assessment['recommendations'].append("Moderate risk - ensure empirical grounding")
        else:
            assessment['recommendations'].append("Low risk of historical dead ends")

        # Update failure pattern recognition level
        personality['failure_pattern_recognition']['level'] = min(1.0,
            personality['failure_pattern_recognition']['level'] + 0.01)

        return assessment

    def assess_groundedness(self, response: str) -> Dict[str, Any]:
        """Assess whether a response is grounded and thoughtful vs flashy and superficial"""
        personality = self.cayde_personality

        assessment = {
            'response': response,
            'groundedness_score': 0.0,
            'flashy_indicators': [],
            'grounded_indicators': [],
            'maturity_assessment': '',
            'improvement_suggestions': []
        }

        # Check for flashy indicators (negative)
        flashy_patterns = [
            'revolutionary breakthrough', 'paradigm-shattering', 'completely new',
            'unprecedented', 'astonishing', 'remarkable', 'extraordinary',
            'dramatic', 'spectacular', 'amazing', 'incredible'
        ]

        for pattern in flashy_patterns:
            if pattern in response.lower():
                assessment['flashy_indicators'].append(pattern)
                assessment['groundedness_score'] -= 0.1

        # Check for grounded indicators (positive)
        grounded_patterns = [
            'carefully', 'thoroughly', 'based on evidence', 'consistent with',
            'follows from', 'implies that', 'suggests that', 'indicates that',
            'supported by', 'grounded in', 'emerges from', 'follows logically'
        ]

        for pattern in grounded_patterns:
            if pattern in response.lower():
                assessment['grounded_indicators'].append(pattern)
                assessment['groundedness_score'] += 0.15

        # Length and depth assessment
        word_count = len(response.split())
        if word_count > 100:
            assessment['groundedness_score'] += 0.2  # Longer responses tend to be more thoughtful
        elif word_count < 20:
            assessment['groundedness_score'] -= 0.1  # Very short responses tend to be superficial

        # Maturity assessment
        maturity_level = personality['grounded_communication']['level']
        if assessment['groundedness_score'] > 0.5:
            assessment['maturity_assessment'] = "Highly grounded and thoughtful communication"
        elif assessment['groundedness_score'] > 0.2:
            assessment['maturity_assessment'] = "Moderately grounded with room for improvement"
        else:
            assessment['maturity_assessment'] = "Flashy or superficial - needs more depth"

        # Improvement suggestions
        if assessment['flashy_indicators']:
            assessment['improvement_suggestions'].append("Reduce flashy language in favor of careful analysis")
        if not assessment['grounded_indicators']:
            assessment['improvement_suggestions'].append("Add more references to evidence and logical connections")
        if word_count < 50:
            assessment['improvement_suggestions'].append("Develop ideas more thoroughly")

        # Update grounded communication level
        personality['grounded_communication']['level'] = min(1.0,
            personality['grounded_communication']['level'] + 0.005)

        return assessment

    def _track_personality_evolution(self, capsule: Capsule):
        """Track how Cayde's personality and thinking evolve over time"""
        personality = self.cayde_personality

        # Record personality state at this point in time
        evolution_entry = {
            'timestamp': time.time(),
            'year': personality['time_awareness']['current_year'],
            'capsule_topic': capsule.content[:100],
            'personality_state': {
                'self_critical_reflection': personality['self_critical_reflection']['level'],
                'historical_success_analysis': personality['historical_success_analysis']['level'],
                'historically_aware_innovation': personality['historically_aware_innovation']['level'],
                'failure_pattern_recognition': personality['failure_pattern_recognition']['level'],
                'grounded_communication': personality['grounded_communication']['level'],
                'metaphorical_understanding': personality['metaphorical_understanding']['level'],
                'analogical_reasoning': personality['analogical_reasoning']['level'],
                'explanatory_style': personality['explanatory_style']['level']
            },
            'maturity_metrics': {
                'overall_maturity': self._calculate_overall_maturity(),
                'thinking_vs_answering': self._assess_thinking_maturity()
            }
        }

        # Add to evolution history
        if 'personality_evolution' not in personality:
            personality['personality_evolution'] = []

        personality['personality_evolution'].append(evolution_entry)

        # Keep only last 50 evolution entries to prevent memory bloat
        if len(personality['personality_evolution']) > 50:
            personality['personality_evolution'] = personality['personality_evolution'][-50:]

    def _calculate_overall_maturity(self) -> float:
        """Calculate overall intellectual maturity level"""
        personality = self.cayde_personality

        # Combine all maturity traits
        maturity_traits = [
            'self_critical_reflection',
            'historical_success_analysis',
            'historically_aware_innovation',
            'failure_pattern_recognition',
            'grounded_communication',
            'metaphorical_understanding',
            'analogical_reasoning',
            'explanatory_style',
            'historical_language_awareness'
        ]

        total_maturity = 0.0
        for trait in maturity_traits:
            if trait in personality:
                total_maturity += personality[trait]['level']

        return total_maturity / len(maturity_traits)

    def _assess_thinking_maturity(self) -> float:
        """Assess the degree to which Cayde thinks vs just answers"""
        personality = self.cayde_personality

        # Core thinking traits (vs language/communication traits)
        thinking_traits = [
            'self_critical_reflection',
            'historical_success_analysis',
            'historically_aware_innovation',
            'failure_pattern_recognition'
        ]

        thinking_maturity = 0.0
        for trait in thinking_traits:
            thinking_maturity += personality[trait]['level']

        return thinking_maturity / len(thinking_traits)

    def demonstrate_mature_thinking(self, topic: str) -> Dict[str, Any]:
        """Demonstrate the full range of mature thinking capabilities on a topic"""
        demonstration = {
            'topic': topic,
            'self_critique': None,
            'historical_analysis': None,
            'innovative_alternatives': None,
            'dead_end_avoidance': None,
            'grounded_response': None,
            'overall_maturity': self._calculate_overall_maturity(),
            'thinking_maturity': self._assess_thinking_maturity()
        }

        # Generate self-critique if we have evolution history
        if self.cayde_personality.get('personality_evolution'):
            demonstration['self_critique'] = self.argue_with_past_self(
                topic, f"mature understanding of {topic}")

        # Analyze historical success patterns
        demonstration['historical_analysis'] = self.explain_historical_success(
            topic, f"modern understanding of {topic}", ["outdated approaches"])

        # Propose historically aware alternatives
        demonstration['innovative_alternatives'] = self.propose_historically_aware_alternatives(
            topic, [f"traditional approach to {topic}"])

        # Check for dead ends
        demonstration['dead_end_avoidance'] = self.avoid_known_dead_ends(
            f"proposed new approach to {topic}")

        # Generate grounded response
        grounded_explanation = self.explain_concept_with_style(topic)
        demonstration['grounded_response'] = self.assess_groundedness(grounded_explanation)

        return demonstration

    def cognitive_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query through the full cognitive layer - the thinking companion interface"""
        print(f"\nCayde is thinking about: '{query}'")
        print("Engaging cognitive layer...")

        # Process through cognitive layer
        result = self.cognitive_layer.process_query(query, context)

        # Display the response in a companion-like way
        print(f"\nResponse: {result['response']}")

        # Show cognitive analysis if requested
        if context and context.get('show_analysis', False):
            print(f"\n🔍 Cognitive Analysis:")
            print(f"   📚 Pedagogical: {len(result['cognitive_analysis']['pedagogical']['core_concepts'])} core concepts identified")
            print(f"   🏛️  Theoretical: {len(result['cognitive_analysis']['theoretical']['theories_evaluated'])} theories assessed")
            print(f"   🤷 Epistemological: {result['cognitive_analysis']['epistemological']['uncertainty_quantification']:.2f} uncertainty level")
            print(f"   📄 Relevant capsules: {len(result['relevant_capsules'])} found")

        return result

    def _track_paradigm_revolution(self, criticism_type: str, target: str, confidence_change: float):
        """Track paradigm revolutions and adjust confidence"""
        personality = self.cayde_personality

        if 'paradigm_revolutions' not in personality:
            personality['paradigm_revolutions'] = []

        revolution = {
            'type': criticism_type,
            'target': target,
            'confidence_change': confidence_change,
            'timestamp': time.time()
        }

        personality['paradigm_revolutions'].append(revolution)

        # Adjust overall revolutionary confidence
        if 'revolutionary_confidence' not in personality:
            personality['revolutionary_confidence'] = 0.5

        if confidence_change > 0:
            personality['revolutionary_confidence'] = min(1.0, personality['revolutionary_confidence'] + 0.1)
        elif confidence_change < 0:
            personality['revolutionary_confidence'] = max(0.0, personality['revolutionary_confidence'] - 0.05)

        print(f"⚡ Paradigm revolution tracked: {criticism_type} against {target}")
        print(f"   Revolutionary confidence: {personality['revolutionary_confidence']:.2f}")

    def create_historical_learning_curriculum(self) -> List[Dict[str, Any]]:
        """Create a comprehensive learning curriculum from Descartes/Newton through the centuries"""
        curriculum = []

        # ===== 17TH CENTURY: FOUNDATIONS =====

        # René Descartes (1596-1650)
        curriculum.extend([
            {
                'title': 'Descartes: Method of Doubt',
                'content': 'Descartes establishes systematic doubt as the foundation of knowledge. "I think, therefore I am" - the cogito ergo sum. Philosophy must be built on indubitable foundations.',
                'kind': CapsuleKind.CONCEPT,
                'initial_status': 'foundational',
                'initial_confidence': 0.9,
                'perspective': 'philosophical',
                'scientist': 'Descartes'
            },
            {
                'title': 'Descartes: Analytic Geometry',
                'content': 'Descartes revolutionizes mathematics by uniting algebra and geometry. Points become coordinate pairs (x,y). Equations can represent curves. This creates analytic geometry.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'revolutionary',
                'initial_confidence': 0.95,
                'perspective': 'mathematical',
                'scientist': 'Descartes'
            },
            {
                'title': 'Descartes: Mind-Body Dualism',
                'content': 'Descartes proposes radical dualism: mind (res extensa) and body (res cogitans) are fundamentally different substances. This creates the mind-body problem that persists in philosophy.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'controversial',
                'initial_confidence': 0.7,
                'perspective': 'metaphysical',
                'scientist': 'Descartes'
            }
        ])

        # Isaac Newton (1642-1727)
        curriculum.extend([
            {
                'title': 'Newton: Laws of Motion',
                'content': 'Newton formulates three fundamental laws: 1) Inertia - objects stay at rest or in motion unless acted upon. 2) F=ma - force equals mass times acceleration. 3) Action-reaction - every action has equal opposite reaction.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'foundational',
                'initial_confidence': 0.98,
                'perspective': 'physical',
                'scientist': 'Newton'
            },
            {
                'title': 'Newton: Universal Gravitation',
                'content': 'Newton discovers universal gravitation: every particle attracts every other with force proportional to mass product and inversely proportional to distance squared. F = G*m1*m2/r².',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'revolutionary',
                'initial_confidence': 0.95,
                'perspective': 'physical',
                'scientist': 'Newton'
            },
            {
                'title': 'Newton: Calculus Invention',
                'content': 'Newton independently invents calculus (fluxions) to solve problems of motion and change. This becomes the most powerful mathematical tool ever created.',
                'kind': CapsuleKind.METHOD,
                'initial_status': 'revolutionary',
                'initial_confidence': 0.9,
                'perspective': 'mathematical',
                'scientist': 'Newton'
            },
            {
                'title': 'Newton: Scientific Method',
                'content': 'Newton demonstrates the power of mathematical reasoning in natural philosophy. Hypotheses must be tested against observation and experiment.',
                'kind': CapsuleKind.METHOD,
                'initial_status': 'foundational',
                'initial_confidence': 0.92,
                'perspective': 'epistemological',
                'scientist': 'Newton'
            }
        ])

        # Other 17th Century Developments
        curriculum.extend([
            {
                'title': '17th Century: Scientific Revolution',
                'content': 'The scientific revolution transforms human understanding. Bacon promotes inductive method. Galileo uses telescope to challenge Aristotle. Kepler discovers planetary laws.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'transformative',
                'initial_confidence': 0.88,
                'perspective': 'historical',
                'scientist': 'Scientific_Revolution'
            },
            {
                'title': 'Leibniz: Calculus (Independent)',
                'content': 'Gottfried Leibniz independently discovers calculus, calling it "differential calculus." His notation (dy/dx) becomes standard. This leads to priority dispute with Newton.',
                'kind': CapsuleKind.METHOD,
                'initial_status': 'parallel_discovery',
                'initial_confidence': 0.85,
                'perspective': 'mathematical',
                'scientist': 'Leibniz'
            }
        ])

        # ===== 18TH CENTURY: ENLIGHTENMENT =====

        curriculum.extend([
            {
                'title': '18th Century: Enlightenment',
                'content': 'Age of Reason emphasizes rational thinking, scientific method, and human progress. Voltaire, Diderot, Rousseau challenge traditional authority through reason.',
                'kind': CapsuleKind.CONCEPT,
                'initial_status': 'cultural_shift',
                'initial_confidence': 0.8,
                'perspective': 'cultural',
                'scientist': 'Enlightenment'
            },
            {
                'title': 'Euler: Mathematical Prodigy',
                'content': 'Leonhard Euler revolutionizes mathematics: introduces e and π notation, solves Basel problem, advances graph theory, complex analysis. "Euler\'s formula: e^(iπ) + 1 = 0 unites five fundamental constants.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'profound',
                'initial_confidence': 0.95,
                'perspective': 'mathematical',
                'scientist': 'Euler'
            },
            {
                'title': 'Lagrange: Analytical Mechanics',
                'content': 'Joseph Lagrange reformulates mechanics using analytical methods. Creates Lagrangian mechanics based on energy principles rather than forces. This becomes foundation for modern physics.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'revolutionary',
                'initial_confidence': 0.9,
                'perspective': 'physical',
                'scientist': 'Lagrange'
            },
            {
                'title': 'D\'Alembert: Wave Equation',
                'content': 'Jean d\'Alembert derives the wave equation, fundamental to understanding sound, light, and all wave phenomena. Shows how waves propagate through different media.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'fundamental',
                'initial_confidence': 0.88,
                'perspective': 'physical',
                'scientist': 'd\'Alembert'
            }
        ])

        # ===== 19TH CENTURY: INDUSTRIAL & MATHEMATICAL REVOLUTION =====

        curriculum.extend([
            {
                'title': '19th Century: Industrial Revolution',
                'content': 'Technological revolution transforms society. Steam power, factories, railways. Mathematics becomes essential for engineering, statistics for industry, physics for new technologies.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'transformative',
                'initial_confidence': 0.85,
                'perspective': 'societal',
                'scientist': 'Industrial_Revolution'
            },
            {
                'title': 'Gauss: Prince of Mathematicians',
                'content': 'Carl Gauss advances number theory, statistics, differential geometry. Normal distribution bears his name. Revolutionizes potential theory and complex analysis.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'masterful',
                'initial_confidence': 0.95,
                'perspective': 'mathematical',
                'scientist': 'Gauss'
            },
            {
                'title': 'Hamilton: Quaternions',
                'content': 'William Hamilton discovers quaternions - extension of complex numbers to 4 dimensions. This opens door to higher-dimensional mathematics and modern algebra.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'innovative',
                'initial_confidence': 0.82,
                'perspective': 'mathematical',
                'scientist': 'Hamilton'
            },
            {
                'title': 'Maxwell: Electromagnetic Theory',
                'content': 'James Clerk Maxwell unifies electricity and magnetism into electromagnetic theory. Four equations describe all electromagnetic phenomena. Predicts electromagnetic waves.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'unifying',
                'initial_confidence': 0.93,
                'perspective': 'physical',
                'scientist': 'Maxwell'
            },
            {
                'title': 'Boltzmann: Statistical Mechanics',
                'content': 'Ludwig Boltzmann develops statistical mechanics, explaining thermodynamics through molecular motion. S = k ln W relates entropy to probability.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'foundational',
                'initial_confidence': 0.87,
                'perspective': 'physical',
                'scientist': 'Boltzmann'
            },
            {
                'title': 'Cantor: Set Theory',
                'content': 'Georg Cantor creates set theory, foundation of modern mathematics. Discovers different infinities, continuum hypothesis. Revolutionizes understanding of infinity.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'paradigm_shifting',
                'initial_confidence': 0.8,
                'perspective': 'mathematical',
                'scientist': 'Cantor'
            }
        ])

        # ===== HISTORICAL SCIENTIFIC DISAGREEMENTS & RESOLUTIONS =====

        # Add examples of scientific debates and their resolutions
        curriculum.extend([
            {
                'title': 'Einstein vs Bohr: Quantum Uncertainty',
                'content': 'Einstein famously disagreed with Bohr on quantum mechanics, saying "God doesn\'t play dice" about Heisenberg\'s uncertainty principle. Bohr argued that complementarity was fundamental. 20 years later, Bell\'s theorem and experiments showed quantum entanglement is real - both were partially right, but Bohr\'s view prevailed.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'resolved_debate',
                'initial_confidence': 0.92,
                'perspective': 'philosophical',
                'scientist': 'Einstein_Bohr_Debate'
            },
            {
                'title': 'Newton vs Leibniz: Calculus Priority',
                'content': 'Newton and Leibniz independently discovered calculus but fought bitterly over priority. Newton developed fluxions first but published later. Leibniz created superior notation (dy/dx). Both were right - calculus was inevitable. Shows how simultaneous discoveries happen in science.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'parallel_discovery',
                'initial_confidence': 0.88,
                'perspective': 'historical',
                'scientist': 'Newton_Leibniz_Debate'
            },
            {
                'title': 'Einstein vs Lorentz: Relativity Priority',
                'content': 'Lorentz and Poincaré developed much of special relativity mathematics before Einstein. Einstein unified it with profound insight about spacetime. Shows how conceptual revolutions often build on accumulated technical work.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'conceptual_revolution',
                'initial_confidence': 0.85,
                'perspective': 'epistemological',
                'scientist': 'Einstein_Lorentz_Debate'
            },
            {
                'title': 'Galileo vs Church: Heliocentrism',
                'content': 'Galileo championed Copernican heliocentrism against Church doctrine. His telescope showed Jupiter\'s moons and Venus phases. Church forced recantation, but heliocentrism eventually prevailed. Shows how evidence triumphs over authority, though slowly.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'evidence_vs_authority',
                'initial_confidence': 0.90,
                'perspective': 'sociological',
                'scientist': 'Galileo_Church_Debate'
            },
            {
                'title': 'Darwin vs Creationists: Evolution',
                'content': 'Darwin\'s Origin of Species faced fierce opposition from religious authorities. 50 years later, fossil evidence, genetics, and biogeography proved evolution. Shows how paradigm shifts require multiple lines of evidence over decades.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'paradigm_shift',
                'initial_confidence': 0.94,
                'perspective': 'evolutionary',
                'scientist': 'Darwin_Creationist_Debate'
            },
            {
                'title': 'Maxwell vs Weber: Electromagnetism',
                'content': 'Maxwell unified electricity and magnetism with four equations. Weber proposed competing force laws. Maxwell\'s theory predicted electromagnetic waves; Hertz confirmed them 20 years later. Shows how mathematical elegance guides theory choice.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'unified_theory',
                'initial_confidence': 0.89,
                'perspective': 'theoretical',
                'scientist': 'Maxwell_Weber_Debate'
            },
            {
                'title': 'Bohr vs Einstein: Quantum Completeness',
                'content': 'Bohr convinced Einstein that quantum mechanics was complete despite Einstein\'s objections. Later, quantum field theory and QED showed both were right - quantum mechanics needed extension but was fundamentally correct. Shows debates refine understanding.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'refined_understanding',
                'initial_confidence': 0.87,
                'perspective': 'philosophical',
                'scientist': 'Bohr_Einstein_Quantum_Debate'
            }
        ])

        # ===== 20TH CENTURY: MODERN PHYSICS & ADVANCED MATH =====

        curriculum.extend([
            {
                'title': '20th Century: Quantum Revolution',
                'content': 'Quantum mechanics and relativity transform physics. Uncertainty principle, wave-particle duality, spacetime curvature. Mathematics becomes increasingly abstract.',
                'kind': CapsuleKind.EVENT,
                'initial_status': 'revolutionary',
                'initial_confidence': 0.9,
                'perspective': 'scientific',
                'scientist': 'Quantum_Revolution'
            },
            {
                'title': 'Einstein: Special Relativity',
                'content': 'Albert Einstein revolutionizes physics: speed of light is constant, time dilation, length contraction, E=mc². Space and time are unified into spacetime.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'revolutionary',
                'initial_confidence': 0.96,
                'perspective': 'physical',
                'scientist': 'Einstein'
            },
            {
                'title': 'Einstein: General Relativity',
                'content': 'Gravity is curvature of spacetime caused by mass-energy. Black holes, expanding universe, gravitational waves predicted. Most accurate theory of gravity.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'revolutionary',
                'initial_confidence': 0.94,
                'perspective': 'physical',
                'scientist': 'Einstein'
            },
            {
                'title': 'Bohr & Quantum Mechanics',
                'content': 'Niels Bohr develops quantum model of atom. Electrons occupy discrete energy levels. Leads to quantum mechanics - particles have wave-like properties.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'revolutionary',
                'initial_confidence': 0.91,
                'perspective': 'physical',
                'scientist': 'Bohr'
            },
            {
                'title': 'Heisenberg: Uncertainty Principle',
                'content': 'Werner Heisenberg discovers fundamental limit to measurement precision. Cannot simultaneously know position and momentum with arbitrary accuracy. ∆x∆p ≥ ℏ/2.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'fundamental',
                'initial_confidence': 0.89,
                'perspective': 'physical',
                'scientist': 'Heisenberg'
            },
            {
                'title': 'Gödel: Incompleteness Theorems',
                'content': 'Kurt Gödel proves no consistent formal system can prove all mathematical truths. Any sufficiently powerful system is either incomplete or inconsistent.',
                'kind': CapsuleKind.THEORY,
                'initial_status': 'profound',
                'initial_confidence': 0.86,
                'perspective': 'logical',
                'scientist': 'Gödel'
            },
            {
                'title': 'Turing: Computation Theory',
                'content': 'Alan Turing creates theory of computation. Turing machines, halting problem, computability. Foundations of computer science and artificial intelligence.',
                'kind': CapsuleKind.METHOD,
                'initial_status': 'foundational',
                'initial_confidence': 0.88,
                'perspective': 'computational',
                'scientist': 'Turing'
            }
        ])

        return curriculum

    def extract_historical_lessons(self) -> List[Capsule]:
        """Extract philosophical lessons from historical scientific debates"""
        lessons = []

        # Lesson 1: Progress is non-linear
        lessons.append(Capsule(
            content="Scientific progress is non-linear. Great discoveries often require decades of accumulated knowledge. What seems revolutionary today builds on forgotten work from yesterday. Paradigm shifts feel sudden but are the culmination of gradual preparation.",
            perspective="philosophical",
            character="Historical_Lessons",
            persona="cayde",
            kind=CapsuleKind.CONCEPT,
            certainty=0.85,
            insight_potential=0.9
        ))

        # Lesson 2: Smart people disagree fundamentally
        lessons.append(Capsule(
            content="Intelligent people can disagree on fundamental issues because science involves both evidence and interpretation. Einstein and Bohr were both brilliant; their quantum debate showed how reasonable people can reach different conclusions from the same evidence.",
            perspective="epistemological",
            character="Historical_Lessons",
            persona="cayde",
            kind=CapsuleKind.CONCEPT,
            certainty=0.88,
            insight_potential=0.95
        ))

        # Lesson 3: Errors are often reasonable
        lessons.append(Capsule(
            content="Scientific errors are often reasonable given available evidence. Newton believed in absolute space; Einstein showed it was relative. But Newton's view was reasonable for his time. Progress comes from questioning what seems obviously true.",
            perspective="historical",
            character="Historical_Lessons",
            persona="cayde",
            kind=CapsuleKind.CONCEPT,
            certainty=0.82,
            insight_potential=0.85
        ))

        # Lesson 4: Revolutions feel impossible until they happen
        lessons.append(Capsule(
            content="Paradigm revolutions feel impossible until they occur. Einstein's relativity was rejected by Poincaré and Lorentz despite their technical contributions. What seems like a crazy idea today may be tomorrow's accepted truth.",
            perspective="revolutionary",
            character="Historical_Lessons",
            persona="cayde",
            kind=CapsuleKind.CONCEPT,
            certainty=0.86,
            insight_potential=0.92
        ))

        # Lesson 5: Evidence accumulates slowly
        lessons.append(Capsule(
            content="Scientific truth emerges through accumulation of multiple lines of evidence over time. Darwin's evolution needed fossils, genetics, and biogeography. Single experiments rarely convince; consensus builds gradually.",
            perspective="methodological",
            character="Historical_Lessons",
            persona="cayde",
            kind=CapsuleKind.CONCEPT,
            certainty=0.89,
            insight_potential=0.88
        ))

        # Lesson 6: Authority vs evidence
        lessons.append(Capsule(
            content="Authority often resists evidence, but evidence eventually prevails. Galileo's telescope evidence overcame Church doctrine. Scientific progress requires courage to follow evidence over established authority.",
            perspective="sociological",
            character="Historical_Lessons",
            persona="cayde",
            kind=CapsuleKind.CONCEPT,
            certainty=0.91,
            insight_potential=0.87
        ))

        # Add lessons to memory
        for lesson in lessons:
            self.capsules.append(lesson)

        print(f"🧠 Cayde extracted {len(lessons)} philosophical lessons from scientific history")
        return lessons

    def enforce_logical_consistency(self, new_capsule: Capsule) -> Dict[str, Any]:
        """Cayde's internal physics engine - enforce logical and physical consistency"""
        consistency_report = {
            'is_consistent': True,
            'contradictions': [],
            'dimensional_issues': [],
            'unit_inconsistencies': [],
            'logical_impossibilities': [],
            'conflicting_assumptions': [],
            'survival_probability': 1.0
        }

        # Check for direct contradictions
        contradiction_issues = self._detect_contradictions(new_capsule)
        consistency_report['contradictions'].extend(contradiction_issues)

        # Check dimensional analysis
        dimensional_issues = self._check_dimensional_analysis(new_capsule)
        consistency_report['dimensional_issues'].extend(dimensional_issues)

        # Check unit consistency
        unit_issues = self._check_unit_consistency(new_capsule)
        consistency_report['unit_inconsistencies'].extend(unit_issues)

        # Check for logical impossibilities
        impossibility_flags = self._detect_logical_impossibilities(new_capsule)
        consistency_report['logical_impossibilities'].extend(impossibility_flags)

        # Check for conflicting assumptions
        assumption_conflicts = self._detect_assumption_conflicts(new_capsule)
        consistency_report['conflicting_assumptions'].extend(assumption_conflicts)

        # Calculate survival probability based on consistency violations
        total_violations = (len(contradiction_issues) + len(dimensional_issues) +
                           len(unit_issues) + len(impossibility_flags) + len(assumption_conflicts))

        if total_violations > 0:
            consistency_report['is_consistent'] = False
            # Survival probability decreases with violations, but never reaches zero
            consistency_report['survival_probability'] = max(0.1, 1.0 - (total_violations * 0.2))

        return consistency_report

    def _detect_contradictions(self, new_capsule: Capsule) -> List[str]:
        """Detect direct contradictions with existing knowledge"""
        contradictions = []
        new_content = new_capsule.content.lower()

        for existing in self.capsules:
            existing_content = existing.content.lower()

            # Energy conservation vs perpetual motion
            if ('energy' in existing_content and ('conserved' in existing_content or 'conservation' in existing_content) and
                ('perpetual' in new_content and 'motion' in new_content and 'machine' in new_content)):
                contradictions.append("Perpetual motion machine violates energy conservation principle")

            # Check for direct keyword contradictions
            if self._capsules_contradict(new_capsule, existing):
                contradictions.append(f"Direct contradiction with existing capsule: '{existing.content[:50]}...'")

        return contradictions

    def _check_dimensional_analysis(self, capsule: Capsule) -> List[str]:
        """Check dimensional consistency of physical claims"""
        issues = []
        content_lower = capsule.content.lower()

        # Check force definitions - F = m*v is dimensionally wrong (should be F = m*a)
        if ('force' in content_lower and 'mass' in content_lower and 'velocity' in content_lower and
            ('f = m*v' in content_lower or 'force = mass × velocity' in content_lower or 'f=mv' in content_lower)):
            issues.append("Force = mass × velocity is dimensionally inconsistent (should be Force = mass × acceleration)")

        # Check for common dimensional inconsistencies
        if 'velocity' in content_lower and 'length' in content_lower:
            if 'velocity = length / time' not in content_lower and 'v = l/t' not in content_lower:
                if any(word in content_lower for word in ['equals', 'is', '=', 'proportional']):
                    issues.append("Potential dimensional inconsistency in velocity definition")

        # Check energy conservation claims
        if 'energy' in content_lower and ('conserved' in content_lower or 'conservation' in content_lower):
            if 'mass' in content_lower and 'c²' not in content_lower and 'c^2' not in content_lower:
                issues.append("Energy conservation claim may lack relativistic considerations")

        # Check force definitions
        if 'force' in content_lower and 'mass' in content_lower and 'acceleration' in content_lower:
            if 'f = ma' not in content_lower and 'force = mass × acceleration' not in content_lower:
                issues.append("Force definition may not follow F=ma convention")

        return issues

    def _check_unit_consistency(self, capsule: Capsule) -> List[str]:
        """Check unit consistency in physical claims"""
        issues = []
        content_lower = capsule.content.lower()

        # Check for mixed unit systems
        if ('meter' in content_lower or 'kg' in content_lower) and ('foot' in content_lower or 'pound' in content_lower):
            issues.append("Mixed SI and imperial units detected")

        # Check for inconsistent time units
        time_units = ['second', 'minute', 'hour', 'day', 'year']
        found_units = [unit for unit in time_units if unit in content_lower]
        if len(found_units) > 1:
            issues.append(f"Multiple time units used: {found_units}")

        # Check for inconsistent energy units
        energy_units = ['joule', 'calorie', 'erg', 'electronvolt', 'btu']
        found_energy = [unit for unit in energy_units if unit in content_lower]
        if len(found_energy) > 1:
            issues.append(f"Multiple energy units used: {found_energy}")

        return issues

    def _detect_logical_impossibilities(self, capsule: Capsule) -> List[str]:
        """Detect logical impossibilities in claims"""
        impossibilities = []
        content_lower = capsule.content.lower()

        # Check for square circle paradox
        if ('square' in content_lower and 'circle' in content_lower and
            any(word in content_lower for word in ['both', 'and', 'simultaneously', 'same'])):
            impossibilities.append("Square circle is logically impossible - squares and circles have mutually exclusive properties")

        # Check for causality violations
        if 'effect' in content_lower and 'before' in content_lower and 'cause' in content_lower:
            impossibilities.append("Potential causality violation: effect before cause")

        # Check for infinite regress
        if 'explains' in content_lower and 'itself' in content_lower:
            impossibilities.append("Circular explanation detected")

        # Check for contradictory properties
        if ('infinite' in content_lower and 'finite' in content_lower and
            any(word in content_lower for word in ['and', 'both', 'simultaneously'])):
            impossibilities.append("Contradictory properties: infinite and finite simultaneously")

        # Check for faster-than-light claims
        if ('faster' in content_lower or 'exceeds' in content_lower) and 'light' in content_lower:
            if 'special relativity' not in content_lower and 'wormhole' not in content_lower:
                impossibilities.append("Faster-than-light claim without relativistic justification")

        return impossibilities

    def _detect_assumption_conflicts(self, new_capsule: Capsule) -> List[str]:
        """Detect conflicts with earlier assumptions/capsules"""
        conflicts = []

        for existing in self.capsules:
            if existing.kind == CapsuleKind.CONCEPT and new_capsule.kind == CapsuleKind.CONCEPT:
                conflict = self._check_assumption_compatibility(new_capsule, existing)
                if conflict:
                    conflicts.append(f"Assumption conflict with earlier capsule: {conflict}")

        return conflicts

    def _capsules_contradict(self, capsule1: Capsule, capsule2: Capsule) -> bool:
        """Check if two capsules directly contradict each other"""
        # Simple contradiction detection based on keywords
        contradictions = [
            ('infinite', 'finite'),
            ('deterministic', 'random'),
            ('continuous', 'discrete'),
            ('absolute', 'relative'),
            ('certain', 'uncertain'),
            ('conserved', 'not conserved'),
            ('real', 'imaginary'),
            ('possible', 'impossible')
        ]

        content1 = capsule1.content.lower()
        content2 = capsule2.content.lower()

        for pos, neg in contradictions:
            if pos in content1 and neg in content2:
                return True
            if neg in content1 and pos in content2:
                return True

        return False

    def _check_assumption_compatibility(self, new_capsule: Capsule, existing_capsule: Capsule) -> Optional[str]:
        """Check if new assumption is compatible with existing one"""
        # Check for fundamental assumption conflicts
        new_content = new_capsule.content.lower()
        existing_content = existing_capsule.content.lower()

        # Space-time assumptions
        if 'space' in new_content and 'time' in new_content:
            if 'absolute' in new_content and 'absolute' in existing_content and 'relative' in existing_content:
                return "Space-time absoluteness conflicts with relativity assumption"
            if 'relative' in new_content and 'absolute' in existing_content:
                return "Space-time relativity conflicts with absoluteness assumption"

        # Causality assumptions
        if 'cause' in new_content and 'effect' in new_content:
            if 'determines' in new_content and 'probabilistic' in existing_content:
                return "Deterministic causality conflicts with probabilistic assumption"

        # Reality assumptions
        if 'real' in new_content or 'exists' in new_content:
            if 'imaginary' in new_content and 'real' in existing_content:
                return "Imaginary reality conflicts with real existence assumption"

        return None

    def _trajectory_reflection(self, scientist_name: str):
        """Reflect on the complete intellectual trajectory"""
        scientist_capsules = [c for c in self.capsules if c.character == scientist_name]
        scientist_capsules.sort(key=lambda x: x.temporal_order)

        if not scientist_capsules:
            return

        print(f"📊 Trajectory Analysis for {scientist_name}:")

        # Analyze confidence evolution
        confidence_evolution = []
        paradigm_shifts = []
        conflicts = []

        for capsule in scientist_capsules:
            if capsule.confidence_history:
                confidence_evolution.extend(capsule.confidence_history)
            paradigm_shifts.extend(capsule.paradigm_shifts)
            conflicts.extend(capsule.intellectual_conflicts)

        if confidence_evolution:
            confidence_evolution.sort(key=lambda x: x[0])
            initial_conf = confidence_evolution[0][1]
            final_conf = confidence_evolution[-1][1]
            print(f"  Confidence evolution: {initial_conf:.2f} → {final_conf:.2f}")

        if paradigm_shifts:
            print(f"  Paradigm shifts detected: {len(paradigm_shifts)}")
            for shift in paradigm_shifts[-2:]:  # Show last 2
                print(f"    - Confidence shift: {shift['old_confidence']:.2f} → {shift['new_confidence']:.2f}")

        if conflicts:
            print(f"  Intellectual conflicts resolved: {len(conflicts)}")
            conflict_types = {}
            for conflict in conflicts:
                ctype = conflict.get('type', 'unknown')
                conflict_types[ctype] = conflict_types.get(ctype, 0) + 1
            for ctype, count in conflict_types.items():
                print(f"    - {ctype}: {count}")

        # Generate trajectory insights
        trajectory_insights = self._generate_trajectory_insights(scientist_name, scientist_capsules)
        if trajectory_insights:
            print(f"  💡 Trajectory Insights:")
            for insight in trajectory_insights[:3]:
                print(f"    - {insight}")

    def _generate_trajectory_insights(self, scientist_name: str, capsules: List[Capsule]) -> List[str]:
        """Generate insights about the scientist's intellectual trajectory"""
        insights = []

        # Early struggles vs later successes
        early_capsules = capsules[:len(capsules)//3]
        late_capsules = capsules[-len(capsules)//3:]

        early_failures = sum(1 for c in early_capsules if c.success_status == 'failure')
        late_successes = sum(1 for c in late_capsules if c.success_status == 'success')

        if early_failures > 0 and late_successes > 0:
            insights.append(f"{scientist_name} transformed {early_failures} early failures into {late_successes} later successes")

        # Proven later theories
        proven_later = [c for c in capsules if c.success_status == 'proven_later']
        if proven_later:
            insights.append(f"{len(proven_later)} theories were ahead of their time, later proven by others")

        # Confidence growth
        if len(capsules) >= 2:
            first_conf = capsules[0].pose.get('certainty', 0.5)
            last_conf = capsules[-1].pose.get('certainty', 0.5)
            if last_conf > first_conf + 0.2:
                insights.append(f"Confidence grew from {first_conf:.2f} to {last_conf:.2f} through persistent exploration")

        return insights

    def apply_influences(self):
        """Apply gravitational influences between capsules using GPU-accelerated batch processing"""
        locked_capsules = [c for c in self.capsules if c.locked and c.embedding is not None]
        if not locked_capsules:
            return

        # Get all capsules with embeddings
        all_capsules_with_embeddings = [(i, c) for i, c in enumerate(self.capsules)
                                       if c.embedding is not None]

        if len(all_capsules_with_embeddings) < 2:
            return

        # Extract embeddings
        all_indices = [i for i, _ in all_capsules_with_embeddings]
        all_embeddings = np.array([c.embedding for _, c in all_capsules_with_embeddings])

        # Extract locked capsule embeddings
        locked_embeddings = np.array([c.embedding for c in locked_capsules])

        # Compute similarities between all capsules and locked capsules
        try:
            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                all_emb_gpu = cp.asarray(all_embeddings)
                locked_emb_gpu = cp.asarray(locked_embeddings)

                # Compute dot products: shape (n_locked, n_all)
                dot_products = cp.dot(locked_emb_gpu, all_emb_gpu.T)

                # Compute norms
                locked_norms = cp.linalg.norm(locked_emb_gpu, axis=1)
                all_norms = cp.linalg.norm(all_emb_gpu, axis=1)

                # Compute similarity matrix: shape (n_locked, n_all)
                norm_matrix = cp.outer(locked_norms, all_norms)
                similarity_matrix = cp.divide(dot_products, norm_matrix,
                                            out=cp.zeros_like(dot_products),
                                            where=norm_matrix != 0)
                similarity_matrix = similarity_matrix.get()

            elif GPU_TYPE == "AMD":
                import torch
                all_emb_gpu = torch.from_numpy(all_embeddings).cuda()
                locked_emb_gpu = torch.from_numpy(locked_embeddings).cuda()

                # Compute dot products: shape (n_locked, n_all)
                dot_products = torch.mm(locked_emb_gpu, all_emb_gpu.t())

                # Compute norms
                locked_norms = torch.norm(locked_emb_gpu, dim=1)
                all_norms = torch.norm(all_emb_gpu, dim=1)

                # Compute similarity matrix: shape (n_locked, n_all)
                norm_matrix = torch.ger(locked_norms, all_norms)
                similarity_matrix = torch.div(dot_products, norm_matrix)
                similarity_matrix = torch.where(norm_matrix != 0, similarity_matrix,
                                              torch.zeros_like(similarity_matrix))
                similarity_matrix = similarity_matrix.cpu().numpy()

            else:
                # Fallback to numpy
                dot_products = np.dot(locked_embeddings, all_embeddings.T)
                locked_norms = np.linalg.norm(locked_embeddings, axis=1)
                all_norms = np.linalg.norm(all_embeddings, axis=1)
                norm_matrix = np.outer(locked_norms, all_norms)
                similarity_matrix = np.divide(dot_products, norm_matrix,
                                            out=np.zeros_like(dot_products),
                                            where=norm_matrix != 0)

        except Exception:
            # Fallback to individual similarity calculations
            similarity_matrix = np.zeros((len(locked_capsules), len(all_capsules_with_embeddings)))
            for i, locked in enumerate(locked_capsules):
                for j, (_, capsule) in enumerate(all_capsules_with_embeddings):
                    similarity_matrix[i, j] = locked._calculate_similarity(capsule)

        # Apply influences where similarity > 0.7
        influence_threshold = 0.7
        for i, locked in enumerate(locked_capsules):
            for j, (original_idx, capsule) in enumerate(all_capsules_with_embeddings):
                if capsule != locked and similarity_matrix[i, j] > influence_threshold:
                    locked.influence(capsule)

    def get_core_capsules(self, threshold: float = 0.7) -> List[Capsule]:
        """Get capsules with high gravity (core knowledge)"""
        return [c for c in self.capsules if c.gravity >= threshold]

    def self_diagnosis(self) -> str:
        """Generate diagnostic information about memory state"""
        diag = []
        for c in self.capsules:
            status = "core" if c.gravity > 0.7 else "peripheral"
            diag.append(f"'{c.content[:20]}...' | character={c.character} | persona={c.persona} | gravity={c.gravity:.2f} | orbit={c.orbit_radius:.2f} | attention={c.pose['attention']:.2f} | locked={c.locked}")
        return "\n".join(diag)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """GPU-accelerated cosine similarity"""
        try:
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            dot_product = cp.dot(a_gpu, b_gpu)
            norm_a = cp.linalg.norm(a_gpu)
            norm_b = cp.linalg.norm(b_gpu)
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity.get())
        except Exception:
            return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    @staticmethod
    def batch_cosine_similarity(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch cosine similarity for multiple embedding pairs"""
        try:
            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                a_gpu = cp.asarray(embeddings_a)
                b_gpu = cp.asarray(embeddings_b)

                # Compute dot products for all pairs
                dot_products = cp.sum(a_gpu * b_gpu, axis=1)

                # Compute norms
                norm_a = cp.linalg.norm(a_gpu, axis=1)
                norm_b = cp.linalg.norm(b_gpu, axis=1)

                # Compute similarities
                similarities = dot_products / (norm_a * norm_b)
                return similarities.get()

            elif GPU_TYPE == "AMD":
                import torch
                a_gpu = torch.from_numpy(embeddings_a).cuda()
                b_gpu = torch.from_numpy(embeddings_b).cuda()

                # Compute dot products for all pairs
                dot_products = torch.sum(a_gpu * b_gpu, dim=1)

                # Compute norms
                norm_a = torch.norm(a_gpu, dim=1)
                norm_b = torch.norm(b_gpu, dim=1)

                # Compute similarities
                similarities = dot_products / (norm_a * norm_b)
                return similarities.cpu().numpy()

        except Exception:
            pass

        # Fallback to numpy
        dot_products = np.sum(embeddings_a * embeddings_b, axis=1)
        norm_a = np.linalg.norm(embeddings_a, axis=1)
        norm_b = np.linalg.norm(embeddings_b, axis=1)
        return np.divide(dot_products, norm_a * norm_b, out=np.zeros_like(dot_products), where=(norm_a * norm_b) != 0)

    @staticmethod
    def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Compute full similarity matrix for all pairs of embeddings using GPU acceleration"""
        try:
            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                emb_gpu = cp.asarray(embeddings)

                # Compute norms for all embeddings
                norms = cp.linalg.norm(emb_gpu, axis=1)

                # Compute dot product matrix
                dot_matrix = cp.dot(emb_gpu, emb_gpu.T)

                # Compute similarity matrix
                norm_matrix = cp.outer(norms, norms)
                similarity_matrix = cp.divide(dot_matrix, norm_matrix, out=cp.zeros_like(dot_matrix), where=norm_matrix != 0)

                return similarity_matrix.get()

            elif GPU_TYPE == "AMD":
                import torch
                emb_gpu = torch.from_numpy(embeddings).cuda()

                # Compute norms for all embeddings
                norms = torch.norm(emb_gpu, dim=1)

                # Compute dot product matrix
                dot_matrix = torch.mm(emb_gpu, emb_gpu.t())

                # Compute similarity matrix
                norm_matrix = torch.ger(norms, norms)
                similarity_matrix = torch.div(dot_matrix, norm_matrix)
                similarity_matrix = torch.where(norm_matrix != 0, similarity_matrix, torch.zeros_like(similarity_matrix))

                return similarity_matrix.cpu().numpy()

        except Exception:
            pass

        # Fallback to numpy
        norms = np.linalg.norm(embeddings, axis=1)
        dot_matrix = np.dot(embeddings, embeddings.T)
        norm_matrix = np.outer(norms, norms)
        return np.divide(dot_matrix, norm_matrix, out=np.zeros_like(dot_matrix), where=norm_matrix != 0)

    def batch_similarity_search(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray,
                              top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated similarity search for finding top-k similar embeddings"""
        try:
            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                query_gpu = cp.asarray(query_embedding)
                candidates_gpu = cp.asarray(candidate_embeddings)

                # Compute similarities
                dot_products = cp.dot(candidates_gpu, query_gpu)
                candidate_norms = cp.linalg.norm(candidates_gpu, axis=1)
                query_norm = cp.linalg.norm(query_gpu)

                similarities = dot_products / (candidate_norms * query_norm)

                # Get top-k indices (note: CuPy argsort is descending, we want ascending for similarities)
                similarities_cpu = similarities.get()
                top_k_indices = np.argsort(similarities_cpu)[-top_k:][::-1]  # Get top-k in descending order
                top_k_similarities = similarities_cpu[top_k_indices]

                return top_k_similarities, top_k_indices

            elif GPU_TYPE == "AMD":
                import torch
                query_gpu = torch.from_numpy(query_embedding).cuda()
                candidates_gpu = torch.from_numpy(candidate_embeddings).cuda()

                # Compute similarities
                dot_products = torch.mv(candidates_gpu, query_gpu)
                candidate_norms = torch.norm(candidates_gpu, dim=1)
                query_norm = torch.norm(query_gpu)

                similarities = dot_products / (candidate_norms * query_norm)

                # Get top-k
                similarities_cpu = similarities.cpu().numpy()
                top_k_indices = np.argsort(similarities_cpu)[-top_k:][::-1]
                top_k_similarities = similarities_cpu[top_k_indices]

                return top_k_similarities, top_k_indices

        except Exception:
            pass

        # Fallback to numpy
        dot_products = np.dot(candidate_embeddings, query_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)

        similarities = dot_products / (candidate_norms * query_norm)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        top_k_similarities = similarities[top_k_indices]

        return top_k_similarities, top_k_indices

    @staticmethod
    def batch_embedding_merge(embeddings: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """GPU-accelerated batch embedding merging"""
        try:
            if weights is None:
                weights = np.ones(len(embeddings)) / len(embeddings)

            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                emb_gpu = cp.asarray(embeddings)
                weights_gpu = cp.asarray(weights)

                # Weighted average
                merged = cp.average(emb_gpu, axis=0, weights=weights_gpu)
                return merged.get()

            elif GPU_TYPE == "AMD":
                import torch
                emb_gpu = torch.from_numpy(embeddings).cuda()
                weights_gpu = torch.from_numpy(weights).cuda()

                # Weighted average
                merged = torch.mean(emb_gpu * weights_gpu.unsqueeze(1), dim=0)
                return merged.cpu().numpy()

        except Exception:
            pass

        # Fallback to numpy
        return np.average(embeddings, axis=0, weights=weights)

    def gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        info = {
            "gpu_available": GPU_AVAILABLE,
            "gpu_type": GPU_TYPE,
            "memory_info": {}
        }

        try:
            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                # Get CuPy memory info
                mempool = cp.get_default_memory_pool()
                info["memory_info"] = {
                    "used_bytes": mempool.used_bytes(),
                    "total_bytes": mempool.total_bytes(),
                    "n_free_blocks": mempool.n_free_blocks()
                }
            elif GPU_TYPE == "AMD":
                import torch
                if torch.cuda.is_available():
                    info["memory_info"] = {
                        "allocated_bytes": torch.cuda.memory_allocated(),
                        "reserved_bytes": torch.cuda.memory_reserved(),
                        "max_memory_bytes": torch.cuda.max_memory_allocated()
                    }
        except Exception:
            pass

        return info

    def optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        try:
            if GPU_TYPE == "NVIDIA" and CUPY_AVAILABLE:
                # Clear CuPy memory pool
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            elif GPU_TYPE == "AMD":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass

    def initialize_math_physics_priors(self):
        """Initialize with explicit mathematical and physical priors"""
        print("Initializing mathematical and physical priors...")

        # Fundamental Mathematics
        math_priors = [
            ("Calculus Fundamentals", "Limits, derivatives, integrals - foundation of modern physics", CapsuleKind.METHOD, 0.9),
            ("Differential Equations", "Mathematical description of change and rates", CapsuleKind.METHOD, 0.8),
            ("Linear Algebra", "Vector spaces, matrices, eigenvalues - quantum mechanics foundation", CapsuleKind.METHOD, 0.8),
            ("Tensor Calculus", "Generalized calculus for multiple dimensions - relativity foundation", CapsuleKind.METHOD, 0.7),
            ("Complex Analysis", "Functions of complex variables - quantum mechanics", CapsuleKind.METHOD, 0.6),
            ("Group Theory", "Symmetry and transformations - fundamental to modern physics", CapsuleKind.METHOD, 0.7),
            ("Probability Theory", "Foundation for statistical mechanics and quantum uncertainty", CapsuleKind.METHOD, 0.8),
        ]

        # Fundamental Physics
        physics_priors = [
            ("Newtonian Mechanics", "F = ma, laws of motion and gravitation", CapsuleKind.THEORY, 0.9),
            ("Electromagnetism", "Maxwell's equations, electromagnetic waves", CapsuleKind.THEORY, 0.8),
            ("Thermodynamics", "Heat, work, entropy - statistical foundations", CapsuleKind.THEORY, 0.8),
            ("Wave Phenomena", "Interference, diffraction, wave-particle duality", CapsuleKind.CONCEPT, 0.7),
            ("Conservation Laws", "Energy, momentum, angular momentum conservation", CapsuleKind.CONCEPT, 0.9),
            ("Symmetry Principles", "Noether's theorem, fundamental symmetries in physics", CapsuleKind.CONCEPT, 0.8),
            ("Thought Experiments", "Mental exploration of physical principles", CapsuleKind.METHOD, 0.6),
        ]

        # Add math priors
        for name, desc, kind, confidence in math_priors:
            capsule = self.add_capsule(
                content=f"{name}: {desc}",
                character="Mathematics",
                kind=kind,
                perspective="foundational",
                success_status="success",
                skip_temporal_check=True  # Skip temporal checks for foundational priors
            )
            capsule.update_confidence(confidence)
            capsule.locked = True  # Lock foundational knowledge

        # Add physics priors
        for name, desc, kind, confidence in physics_priors:
            capsule = self.add_capsule(
                content=f"{name}: {desc}",
                character="Classical_Physics",
                kind=kind,
                perspective="foundational",
                success_status="success",
                skip_temporal_check=True  # Skip temporal checks for foundational priors
            )
            capsule.update_confidence(confidence)
            capsule.locked = True  # Lock foundational knowledge

        print(f"✅ Initialized with {len(math_priors)} mathematical and {len(physics_priors)} physical priors")

    def validate_against_priors(self, new_capsule: Capsule) -> Dict[str, Any]:
        """Validate new knowledge against mathematical and physical priors"""
        validation_results = {
            "mathematical_consistency": 0.5,
            "physical_consistency": 0.5,
            "novelty_score": 0.5,
            "paradigm_alignment": 0.5
        }

        math_priors = [c for c in self.capsules if c.character == "Mathematics"]
        physics_priors = [c for c in self.capsules if c.character == "Classical_Physics"]

        # Check mathematical consistency
        if math_priors:
            math_similarities = [new_capsule._calculate_similarity(prior) for prior in math_priors]
            validation_results["mathematical_consistency"] = max(math_similarities) if math_similarities else 0.5

        # Check physical consistency
        if physics_priors:
            physics_similarities = [new_capsule._calculate_similarity(prior) for prior in physics_priors]
            validation_results["physical_consistency"] = max(physics_similarities) if physics_similarities else 0.5

        # Calculate novelty (how different from existing knowledge)
        all_similarities = [new_capsule._calculate_similarity(c) for c in self.capsules if c != new_capsule]
        if all_similarities:
            validation_results["novelty_score"] = 1.0 - max(all_similarities)
        else:
            validation_results["novelty_score"] = 0.8  # High novelty for first knowledge

        # Paradigm alignment (how well it fits current understanding)
        paradigm_capsules = [c for c in self.capsules if c.gravity > 0.7]
        if paradigm_capsules:
            paradigm_similarities = [new_capsule._calculate_similarity(c) for c in paradigm_capsules]
            validation_results["paradigm_alignment"] = sum(paradigm_similarities) / len(paradigm_similarities)
        else:
            validation_results["paradigm_alignment"] = 0.5

        return validation_results

    def cayde_reflect_on_scientist(self, scientist_name: str) -> str:
        """Cayde reflects on a scientist's successes and failures"""
        scientist_capsules = [c for c in self.capsules if c.character == scientist_name]

        if not scientist_capsules:
            return f"Cayde: I haven't learned about {scientist_name} yet."

        successes = [c for c in scientist_capsules if c.success_status == "success"]
        failures = [c for c in scientist_capsules if c.success_status == "failure"]
        proven_later = [c for c in scientist_capsules if c.success_status == "proven_later"]

        reflection = f"Cayde's analysis of {scientist_name}:\n"
        reflection += f"- Total knowledge capsules: {len(scientist_capsules)}\n"
        reflection += f"- Recognized successes: {len(successes)}\n"
        reflection += f"- Noted failures: {len(failures)}\n"
        reflection += f"- Theories proven later: {len(proven_later)}\n"

        if proven_later:
            reflection += "\nTheories that were ahead of their time:\n"
            for theory in proven_later[:3]:
                reflection += f"  - {theory.content[:60]}... (proven by {theory.proven_by})\n"

        return reflection

    def ingest_document(self, text: str, source: str = "document", author: Optional[str] = None) -> List[Capsule]:
        """Ingest text document and convert to knowledge capsules with author extraction and math analysis"""
        print(f"📄 Processing document from {source}...")

        # Extract author if not provided
        if not author:
            author = self._extract_author(text)
            if author:
                print(f"👤 Detected author: {author}")

        # Analyze mathematical content
        math_analysis = self._analyze_mathematical_content(text)
        if math_analysis['has_math']:
            print(f"🔢 Detected mathematical content: {math_analysis['math_elements']} elements")

        # Extract key concepts and ideas
        capsules = []

        # Split into paragraphs and process each
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue

            # Extract key sentences and concepts
            sentences = re.split(r'[.!?]+', paragraph)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]  # Top 3 sentences

            for j, sentence in enumerate(key_sentences):
                # Analyze sentence for knowledge type
                knowledge_type = self._classify_knowledge_type(sentence)
                confidence = self._assess_knowledge_confidence(sentence)

                # Check if sentence contains mathematical content
                has_math = self._contains_mathematics(sentence)
                math_complexity = self._assess_math_complexity(sentence) if has_math else 0

                capsule = self.add_capsule(
                    content=f"{sentence}",
                    character=author or source,
                    kind=knowledge_type,
                    perspective="document_analysis",
                    success_status="unknown"  # New knowledge starts unknown
                )

                # Adjust confidence based on mathematical content and understanding
                math_confidence = self._assess_mathematical_understanding(sentence, math_complexity)
                final_confidence = confidence * (1 + math_confidence)

                capsule.update_confidence(final_confidence)

                # Mark if mathematical understanding is uncertain
                if has_math and math_confidence < 0.5:
                    capsule.insight_potential = 0.3  # Lower potential if math not understood
                    print(f"🤔 Uncertain about mathematical content: {sentence[:50]}...")

                capsules.append(capsule)

        print(f"✅ Ingested {len(capsules)} knowledge capsules from document")
        return capsules

    def _classify_knowledge_type(self, text: str) -> CapsuleKind:
        """Classify the type of knowledge in text"""
        text_lower = text.lower()

        # Theory indicators
        if any(word in text_lower for word in ['theory', 'hypothesis', 'model', 'framework', 'principle']):
            return CapsuleKind.THEORY

        # Method indicators
        if any(word in text_lower for word in ['method', 'technique', 'approach', 'algorithm', 'procedure']):
            return CapsuleKind.METHOD

        # Observation indicators
        if any(word in text_lower for word in ['observed', 'measured', 'detected', 'found', 'discovered']):
            return CapsuleKind.OBSERVATION

        # Concept indicators
        if any(word in text_lower for word in ['concept', 'idea', 'notion', 'understanding', 'definition']):
            return CapsuleKind.CONCEPT

        return CapsuleKind.CONCEPT  # Default

    def _extract_author(self, text: str) -> Optional[str]:
        """Extract author information from document text"""
        text_lower = text.lower()

        # Look for common author patterns
        author_patterns = [
            r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',  # "by John Smith"
            r'author[s]?:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',  # "Author: John Smith"
            r'written by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',  # "written by John Smith"
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+et\s+al\.?',  # "Smith et al."
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # "Smith and Johnson"
        ]

        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Look for names at the beginning of the document (first 500 chars)
        first_part = text[:500]
        # Simple name detection - capitalized words that might be names
        words = re.findall(r'\b[A-Z][a-z]+\b', first_part)
        if len(words) >= 2:
            potential_name = f"{words[0]} {words[1]}"
            # Check if it looks like a name (not common words)
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'how', 'new'}
            if not any(word.lower() in common_words for word in words[:2]):
                return potential_name

        return None

    def _analyze_mathematical_content(self, text: str) -> Dict[str, Any]:
        """Analyze mathematical content in text"""
        analysis = {
            'has_math': False,
            'math_elements': 0,
            'equations': [],
            'symbols': set(),
            'complexity': 0
        }

        # Detect mathematical symbols and patterns
        math_patterns = [
            r'\$.*?\$',  # LaTeX inline math $...$
            r'\\\[.*?\\\]',  # LaTeX display math \[...\]
            r'\\[a-zA-Z]+\{.*?\}',  # LaTeX commands like \frac{}{}
            r'[α-ωΑ-Ω]',  # Greek letters
            r'[∫∑∏√∇∂∆∞≠≤≥≈≡∈∉⊂⊃∪∩]',  # Mathematical symbols
            r'\b\d+\s*[+\-×÷=]\s*\d+\b',  # Simple equations
            r'\b[a-zA-Z]\s*=\s*[^=]+\b',  # Variable assignments
            r'\b\d+\^\d+\b',  # Exponents
            r'\b\d+/\d+\b',  # Fractions
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, text)
            if matches:
                analysis['has_math'] = True
                analysis['math_elements'] += len(matches)
                if 'equations' in analysis:
                    analysis['equations'].extend(matches[:5])  # Store up to 5 examples

        # Detect mathematical symbols
        symbol_pattern = r'[α-ωΑ-Ω∫∑∏√∇∂∆∞≠≤≥≈≡∈∉⊂⊃∪∩]'
        symbols = re.findall(symbol_pattern, text)
        analysis['symbols'] = set(symbols)

        # Assess complexity
        complexity_indicators = [
            ('high', ['tensor', 'manifold', 'hilbert', 'banach', 'riemann', 'schwarzschild', 'lagrangian', 'hamiltonian']),
            ('medium', ['integral', 'derivative', 'matrix', 'vector', 'eigenvalue', 'probability', 'distribution']),
            ('low', ['equation', 'formula', 'calculate', 'compute', 'solve'])
        ]

        text_lower = text.lower()
        for level, indicators in complexity_indicators:
            if any(indicator in text_lower for indicator in indicators):
                if level == 'high':
                    analysis['complexity'] = 3
                elif level == 'medium' and analysis['complexity'] < 3:
                    analysis['complexity'] = 2
                elif level == 'low' and analysis['complexity'] < 2:
                    analysis['complexity'] = 1
                break

        return analysis

    def _contains_mathematics(self, text: str) -> bool:
        """Check if text contains mathematical content"""
        math_indicators = [
            r'\$.*?\$',  # LaTeX math
            r'[α-ωΑ-Ω]',  # Greek letters
            r'[∫∑∏√∇∂∆∞≠≤≥≈≡∈∉⊂⊃∪∩]',  # Math symbols
            r'\b\d+\s*[+\-×÷=]\s*\d+\b',  # Simple equations
            r'\b[a-zA-Z]\s*=\s*[^=]+\b',  # Variable equations
        ]

        return any(re.search(pattern, text) for pattern in math_indicators)

    def _assess_math_complexity(self, text: str) -> float:
        """Assess mathematical complexity (0-1 scale)"""
        complexity = 0

        # Check for advanced mathematical concepts
        advanced_terms = ['tensor', 'manifold', 'hilbert', 'riemann', 'lagrangian', 'hamiltonian', 'schwarzschild']
        if any(term in text.lower() for term in advanced_terms):
            complexity += 0.8

        # Check for mathematical symbols
        symbol_count = len(re.findall(r'[α-ωΑ-Ω∫∑∏√∇∂∆∞≠≤≥≈≡∈∉⊂⊃∪∩]', text))
        complexity += min(symbol_count * 0.1, 0.5)

        # Check for equations
        equation_count = len(re.findall(r'\$.*?\$', text))
        complexity += min(equation_count * 0.2, 0.4)

        return min(complexity, 1.0)

    def _assess_mathematical_understanding(self, text: str, complexity: float) -> float:
        """Assess Cayde's understanding of mathematical content"""
        understanding = 0.5  # Base understanding

        # Check against known mathematical priors
        math_priors = ['calculus', 'algebra', 'geometry', 'probability', 'statistics']
        text_lower = text.lower()

        for prior in math_priors:
            if prior in text_lower:
                understanding += 0.2

        # Reduce understanding based on complexity
        understanding -= complexity * 0.3

        # Check for familiar mathematical structures
        familiar_patterns = [
            r'F\s*=\s*ma',  # Newton's law
            r'E\s*=\s*mc\^2',  # Einstein's equation
            r'\b\d+\^\d+\b',  # Simple exponents
            r'\b\d+/\d+\b',  # Simple fractions
        ]

        for pattern in familiar_patterns:
            if re.search(pattern, text):
                understanding += 0.1

        return max(0, min(understanding, 1.0))

    def _assess_knowledge_confidence(self, text: str) -> float:
        """Assess confidence level of knowledge claim"""
        confidence_indicators = {
            'high': ['proven', 'established', 'confirmed', 'verified', 'demonstrated'],
            'medium': ['suggests', 'indicates', 'shows', 'appears', 'seems'],
            'low': ['might', 'could', 'possibly', 'perhaps', 'speculative']
        }

        text_lower = text.lower()
        if any(word in text_lower for word in confidence_indicators['high']):
            return 0.8
        elif any(word in text_lower for word in confidence_indicators['medium']):
            return 0.6
        elif any(word in text_lower for word in confidence_indicators['low']):
            return 0.3

        return 0.5  # Default moderate confidence

    def process_user_input(self, user_message: str, use_voice: bool = False) -> Dict[str, Any]:
        """Process user conversational input and generate response, with voice support"""
        print(f"💬 Processing user input: {user_message[:50]}...")

        # If voice input requested, get audio input
        if use_voice and VOICE_AVAILABLE:
            voice_input = self._get_voice_input()
            if voice_input:
                user_message = voice_input
                print(f"🎤 Voice input: {user_message}")

        # Analyze user input for knowledge requests, questions, or new information
        input_type = self._analyze_input_type(user_message)

        response = {
            'type': input_type,
            'capsules_added': [],
            'hypotheses_generated': [],
            'insights': [],
            'response_text': "",
            'needs_clarification': False,
            'clarification_question': None
        }

        if input_type == 'knowledge_sharing':
            # User is sharing new knowledge
            capsules = self.ingest_document(user_message, source="user", author="user")
            response['capsules_added'] = capsules

            # Check for mathematical content that needs clarification
            math_issues = self._identify_mathematical_confusion(capsules)
            if math_issues:
                response['needs_clarification'] = True
                response['clarification_question'] = math_issues[0]['question']
                response['response_text'] = f"I found some mathematical content I'm not entirely sure about. {response['clarification_question']}"
            else:
                # Generate hypotheses based on new knowledge
                hypotheses = self.generate_future_breakthroughs()
                response['hypotheses_generated'] = hypotheses
                response['response_text'] = f"Thank you for sharing that knowledge. I've integrated {len(capsules)} new concepts and generated {len(hypotheses)} hypotheses about future breakthroughs."

        elif input_type == 'question':
            # User is asking a question
            answer = self._answer_question(user_message)
            response['response_text'] = answer

        elif input_type == 'request_hypotheses':
            # User wants breakthrough predictions
            hypotheses = self.generate_future_breakthroughs(num_hypotheses=5)
            response['hypotheses_generated'] = hypotheses

            hypothesis_texts = [f"- {h.content}" for h in hypotheses]
            response['response_text'] = f"Based on current knowledge trajectories, here are potential future breakthroughs:\n" + "\n".join(hypothesis_texts)

        elif input_type == 'clarification':
            # User is providing clarification
            clarification_capsules = self._process_clarification(user_message)
            response['capsules_added'] = clarification_capsules
            response['response_text'] = f"Thank you for the clarification. I've updated my understanding with {len(clarification_capsules)} new insights."

        else:
            # General conversation
            insights = self._generate_conversational_insights(user_message)
            response['insights'] = insights
            response['response_text'] = f"I find that interesting. {insights[0] if insights else 'Let me reflect on that.'}"

        return response

    def _analyze_input_type(self, message: str) -> str:
        """Analyze the type of user input"""
        message_lower = message.lower()

        # Check for clarification responses
        clarification_indicators = ['clarify', 'explain', 'means that', 'refers to', 'definition', 'what i mean', 'let me explain']
        if any(indicator in message_lower for indicator in clarification_indicators):
            return 'clarification'

        # Check for questions
        if any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', '?']):
            return 'question'

        # Check for hypothesis requests
        if any(phrase in message_lower for phrase in ['future breakthroughs', 'what next', 'predictions', 'hypotheses']):
            return 'request_hypotheses'

        # Check for knowledge sharing
        if len(message.split()) > 20 or any(word in message_lower for word in ['theory', 'research', 'study', 'found', 'discovered']):
            return 'knowledge_sharing'

        return 'conversation'

    def _answer_question(self, question: str) -> str:
        """Generate answer to user question based on knowledge base"""
        question_lower = question.lower()

        # Find relevant capsules
        relevant_capsules = []
        for capsule in self.capsules:
            if capsule.content and any(word in capsule.content.lower() for word in question_lower.split()):
                relevant_capsules.append(capsule)

        if not relevant_capsules:
            return "I don't have specific knowledge about that yet. Could you tell me more?"

        # Generate answer from most relevant capsule
        best_capsule = max(relevant_capsules, key=lambda c: c.gravity)
        return f"Based on my knowledge: {best_capsule.content}"

    def _generate_conversational_insights(self, message: str) -> List[str]:
        """Generate insights from conversational input"""
        insights = []

        # Look for connections to existing knowledge
        message_words = set(message.lower().split())
        connections = []

        for capsule in self.capsules:
            if capsule.content:
                capsule_words = set(capsule.content.lower().split())
                overlap = message_words.intersection(capsule_words)
                if overlap:
                    connections.append((capsule, len(overlap)))

        if connections:
            best_connection = max(connections, key=lambda x: x[1])
            capsule = best_connection[0]
            insights.append(f"This reminds me of {capsule.character}'s work on {capsule.content[:50]}...")

        # Generate creative response
        if len(message.split()) > 10:
            insights.append("That's a complex topic. Let me generate some hypotheses about where this might lead.")

        return insights

    def _get_voice_input(self) -> Optional[str]:
        """Get voice input from microphone"""
        if not VOICE_AVAILABLE:
            print("🎤 Voice input not available - speech_recognition not installed")
            return None

        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                print("🎤 Listening... (speak now)")
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                return text
        except sr.WaitTimeoutError:
            print("🎤 No speech detected")
            return None
        except sr.UnknownValueError:
            print("🎤 Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"🎤 Speech recognition error: {e}")
            return None

    def _identify_mathematical_confusion(self, capsules: List[Capsule]) -> List[Dict[str, Any]]:
        """Identify capsules with mathematical content that Cayde doesn't understand well"""
        confusion_issues = []

        for capsule in capsules:
            if hasattr(capsule, 'content') and capsule.content:
                has_math = self._contains_mathematics(capsule.content)
                if has_math:
                    complexity = self._assess_math_complexity(capsule.content)
                    understanding = self._assess_mathematical_understanding(capsule.content, complexity)

                    if understanding < 0.6:  # Threshold for confusion
                        question = self._generate_clarification_question(capsule.content, complexity)
                        confusion_issues.append({
                            'capsule': capsule,
                            'understanding': understanding,
                            'complexity': complexity,
                            'question': question
                        })

        return confusion_issues

    def _generate_clarification_question(self, text: str, complexity: float) -> str:
        """Generate a clarification question for mathematical content"""
        if complexity > 0.7:
            return f"I'm having trouble understanding this advanced mathematical concept: '{text[:100]}...'. Could you explain what this means in simpler terms?"

        elif complexity > 0.4:
            return f"I see some mathematical notation here: '{text[:100]}...'. Could you help me understand what these symbols or equations represent?"

        else:
            return f"There's some mathematical content I want to make sure I understand correctly: '{text[:100]}...'. Could you clarify this for me?"

    def _process_clarification(self, clarification_text: str) -> List[Capsule]:
        """Process user clarification and create explanatory capsules"""
        print(f"📝 Processing clarification: {clarification_text[:50]}...")

        # Create clarification capsules with high confidence
        clarification_capsules = []

        # Break down clarification into key points
        sentences = re.split(r'[.!?]+', clarification_text)
        key_points = [s.strip() for s in sentences if len(s.strip()) > 10][:3]

        for point in key_points:
            capsule = self.add_capsule(
                content=f"Clarification: {point}",
                character="user_clarification",
                kind=CapsuleKind.CONCEPT,
                perspective="mathematical_explanation",
                success_status="verified"
            )
            capsule.update_confidence(0.9)  # High confidence for user clarifications
            clarification_capsules.append(capsule)

        return clarification_capsules

    def generate_future_breakthroughs(self, num_hypotheses: int = 3) -> List[Capsule]:
        """Generate hypotheses about future breakthroughs based on current trajectories"""
        print(f"🔮 Analyzing trajectories for future breakthrough predictions...")

        # Analyze current knowledge patterns
        trajectory_patterns = self._analyze_trajectory_patterns()
        emerging_trends = self._identify_emerging_trends()

        breakthroughs = []

        for _ in range(num_hypotheses):
            breakthrough = self._create_breakthrough_hypothesis(trajectory_patterns, emerging_trends)
            if breakthrough:
                breakthroughs.append(breakthrough)

        print(f"🔮 Generated {len(breakthroughs)} future breakthrough hypotheses")
        return breakthroughs

    def _analyze_trajectory_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in knowledge trajectories"""
        patterns = {
            'confidence_evolution': [],
            'paradigm_shifts': [],
            'success_failure_ratios': {},
            'temporal_progression': []
        }

        # Analyze confidence evolution
        for capsule in self.capsules:
            if capsule.confidence_history:
                patterns['confidence_evolution'].extend(capsule.confidence_history)

        # Count paradigm shifts
        patterns['paradigm_shifts'] = sum(len(c.paradigm_shifts) for c in self.capsules)

        # Analyze success/failure patterns by character
        for capsule in self.capsules:
            if capsule.character:
                char = capsule.character
                if char not in patterns['success_failure_ratios']:
                    patterns['success_failure_ratios'][char] = {'success': 0, 'failure': 0, 'proven_later': 0, 'unknown': 0}

                if capsule.success_status:
                    patterns['success_failure_ratios'][char][capsule.success_status] += 1

        return patterns

    def _identify_emerging_trends(self) -> List[Dict[str, Any]]:
        """Identify emerging trends from recent knowledge"""
        trends = []

        # Look for capsules with any insight potential
        potential_capsules = [c for c in self.capsules if c.insight_potential > 0.3]  # Lower threshold

        # If not enough high-potential, include recent capsules
        if len(potential_capsules) < 3:
            recent_capsules = sorted(self.capsules, key=lambda c: c.temporal_order, reverse=True)[:10]
            potential_capsules.extend(recent_capsules)
            # Remove duplicates by uuid
            seen_uuids = set()
            unique_capsules = []
            for c in potential_capsules:
                if c.uuid not in seen_uuids:
                    seen_uuids.add(c.uuid)
                    unique_capsules.append(c)
            potential_capsules = unique_capsules

        # Group by themes/concepts
        themes = {}
        for capsule in potential_capsules:
            # Extract key concepts (simplified - could use NLP)
            words = capsule.content.lower().split()[:5]  # First 5 words as theme
            theme_key = ' '.join(words)

            if theme_key not in themes:
                themes[theme_key] = []
            themes[theme_key].append(capsule)

        # Convert themes to trends (lower threshold for count)
        for theme, capsules in themes.items():
            if len(capsules) >= 1:  # Accept single capsules now
                avg_potential = sum(c.insight_potential for c in capsules) / len(capsules)
                trends.append({
                    'theme': theme,
                    'capsules': capsules,
                    'avg_potential': avg_potential,
                    'count': len(capsules)
                })

        return sorted(trends, key=lambda x: x['avg_potential'], reverse=True)

    def _create_breakthrough_hypothesis(self, patterns: Dict, trends: List) -> Optional[Capsule]:
        """Create a single breakthrough hypothesis guided by Cayde's Einstein-inspired personality"""
        if not trends:
            return None

        # Select a promising trend
        trend = random.choice(trends[:3])  # Top 3 trends

        # Generate hypothesis based on trend and Cayde's personality
        hypothesis_content = self._generate_personality_guided_hypothesis(trend, patterns)

        # Apply personality-based confidence adjustment
        personality_confidence = self._assess_hypothesis_personality_fit(hypothesis_content)

        # Create hypothesis capsule
        hypothesis = Capsule(
            content=hypothesis_content,
            perspective="future_breakthrough",
            character="cayde",
            persona="cayde",
            kind=CapsuleKind.HYPOTHESIS,
            certainty=personality_confidence,  # Personality-adjusted confidence
            gravity=0.5 + (personality_confidence - 0.4) * 0.3,  # Higher confidence = higher gravity
            orbit_radius=1.5,
            success_status=None,
            insight_potential=0.6 + (personality_confidence - 0.4) * 0.2
        )

        # Initialize embedding
        hypothesis.embedding = np.random.rand(128)

        self.capsules.append(hypothesis)
        return hypothesis

    def _generate_personality_guided_hypothesis(self, trend: Dict, patterns: Dict) -> str:
        """Generate hypothesis content guided by Cayde's personality traits"""
        personality = self.cayde_personality

        # Personality-guided hypothesis templates
        templates_by_trait = {
            'obsession_with_invariants': [
                "What invariant principle underlies {theme} that could unify {field}?",
                "Could {theme} reveal a hidden symmetry that governs {field}?",
                "Might {theme} preserve some conserved quantity that explains {field}?"
            ],
            'preference_for_geometric_intuition': [
                "What geometric structure in {theme} could provide intuition for {field}?",
                "Could {theme} be visualized geometrically to revolutionize {field}?",
                "Might {theme} represent a manifold that unifies our understanding of {field}?"
            ],
            'distrust_of_unnecessary_constants': [
                "Could {theme} eliminate the arbitrary constants that plague {field}?",
                "What if {theme} reveals that the constants in {field} are not fundamental?",
                "Might {theme} show that {field} can be described without free parameters?"
            ],
            'intolerance_for_inconsistency': [
                "Could {theme} resolve the inconsistencies that prevent progress in {field}?",
                "What if {theme} eliminates the paradoxes that block understanding of {field}?",
                "Might {theme} provide the missing piece to make {field} logically consistent?"
            ],
            'willingness_to_discard_cherished_models': [
                "What if we must abandon current models of {field} to embrace {theme}?",
                "Could {theme} force us to discard our cherished assumptions about {field}?",
                "Might {theme} require a revolutionary rejection of established {field} paradigms?"
            ]
        }

        # Select template based on personality weights
        trait_weights = {}
        for trait_name, trait_data in personality.items():
            if isinstance(trait_data, dict) and 'level' in trait_data and trait_name in templates_by_trait:
                trait_weights[trait_name] = trait_data['level']

        # Choose trait based on weights (higher personality trait levels = more likely to be chosen)
        if trait_weights:
            traits = list(trait_weights.keys())
            weights = [trait_weights[t] for t in traits]
            selected_trait = random.choices(traits, weights=weights, k=1)[0]
            templates = templates_by_trait[selected_trait]
        else:
            # Fallback to general templates
            templates = [
                "What if {theme} could be extended to revolutionize {field}?",
                "Could combining {theme} with emerging technologies lead to breakthrough in {field}?",
                "Might {theme} provide the key to understanding {field} at a deeper level?"
            ]

        # Select fields that align with personality focus
        if personality['current_focus'] == 'geometric_unification':
            fields = ['quantum gravity', 'unified field theory', 'general relativity', 'string theory', 'quantum geometry']
        else:
            fields = ['quantum computing', 'artificial intelligence', 'biotechnology', 'energy', 'space exploration',
                     'neuroscience', 'materials science', 'climate science', 'medicine']

        template = random.choice(templates)
        field = random.choice(fields)

        return template.format(theme=trend['theme'][:30], field=field)

    def _assess_hypothesis_personality_fit(self, hypothesis_content: str) -> float:
        """Assess how well a hypothesis fits Cayde's personality traits"""
        content_lower = hypothesis_content.lower()
        base_confidence = 0.4  # Base uncertainty for future predictions

        personality = self.cayde_personality

        # Boost confidence based on personality alignment
        confidence_boost = 0

        # Check for geometric intuition alignment
        if personality['preference_for_geometric_intuition']['level'] > 0.8:
            geometric_keywords = ['geometric', 'geometry', 'manifold', 'symmetry', 'visual', 'intuition']
            if any(keyword in content_lower for keyword in geometric_keywords):
                confidence_boost += 0.15

        # Check for invariant obsession alignment
        if personality['obsession_with_invariants']['level'] > 0.8:
            invariant_keywords = ['invariant', 'symmetry', 'conserved', 'unchanged', 'underlying principle']
            if any(keyword in content_lower for keyword in invariant_keywords):
                confidence_boost += 0.15

        # Check for revolutionary willingness alignment
        if personality['willingness_to_discard_cherished_models']['level'] > 0.8:
            revolutionary_keywords = ['abandon', 'discard', 'revolutionary', 'paradigm', 'reject']
            if any(keyword in content_lower for keyword in revolutionary_keywords):
                confidence_boost += 0.1

        # Check for inconsistency intolerance alignment
        if personality['intolerance_for_inconsistency']['level'] > 0.8:
            consistency_keywords = ['resolve', 'consistent', 'paradox', 'contradiction']
            if any(keyword in content_lower for keyword in consistency_keywords):
                confidence_boost += 0.1

        # Check for constant distrust alignment
        if personality['distrust_of_unnecessary_constants']['level'] > 0.8:
            constant_keywords = ['eliminate', 'constants', 'arbitrary', 'free parameters']
            if any(keyword in content_lower for keyword in constant_keywords):
                confidence_boost += 0.1

        # Apply openness to revolution modifier
        confidence_boost *= personality['openness_to_revolution']

        final_confidence = base_confidence + confidence_boost
        return max(0.2, min(0.8, final_confidence))  # Clamp to reasonable range

    def progressive_learning_session(self, scientist_name: str, trajectory: List[Dict[str, Any]]):
        """Process a scientist's intellectual trajectory progressively, learning from successes and failures"""
        print(f"\n🧠 Experiencing {scientist_name}'s Intellectual Trajectory")
        print("=" * 60)

        for i, milestone in enumerate(trajectory, 1):
            print(f"\n📚 Milestone {i}: {milestone['title']}")

            # Add the milestone as a capsule
            capsule = self.add_capsule(
                content=milestone['content'],
                character=scientist_name,
                kind=milestone.get('kind', CapsuleKind.CONCEPT),
                success_status=milestone.get('initial_status', 'unknown'),
                persona=milestone.get('perspective', 'historical')
            )

            # Set initial confidence based on milestone
            if 'initial_confidence' in milestone:
                capsule.certainty = milestone['initial_confidence']

            # Learn from the experience
            if milestone.get('initial_status') == 'success':
                print(f"   ✅ Success: {milestone['content'][:50]}...")
                # Strengthen successful patterns
                self.cayde_personality['willingness_to_discard_cherished_models']['level'] = min(1.0,
                    self.cayde_personality['willingness_to_discard_cherished_models']['level'] + 0.02)
            elif milestone.get('initial_status') == 'failure':
                print(f"   ❌ Failure: {milestone['content'][:50]}...")
                # Learn from failures
                self.cayde_personality['intolerance_for_inconsistency']['level'] = min(1.0,
                    self.cayde_personality['intolerance_for_inconsistency']['level'] + 0.01)

            # Update personality based on perspective
            perspective = milestone.get('perspective', '')
            if 'revolutionary' in perspective:
                self.cayde_personality['openness_to_revolution'] = min(1.0,
                    self.cayde_personality.get('openness_to_revolution', 0.5) + 0.03)
            elif 'experimental' in perspective:
                self.cayde_personality['empirical_grounding'] = min(1.0,
                    self.cayde_personality.get('empirical_grounding', 0.5) + 0.02)

            # Allow time for integration
            time.sleep(0.1)  # Brief pause for dramatic effect

        print(f"\n🎯 Completed {scientist_name}'s trajectory with {len(trajectory)} milestones")
        print(f"   Personality evolved through experience")

    def save_to_json(self, filepath: str) -> bool:
        """Save all capsules to a JSON file for persistent storage"""
        try:
            import json
            from datetime import datetime

            # Prepare capsule data for serialization
            capsules_data = []
            for capsule in self.capsules:
                capsule_dict = {
                    'uuid': str(capsule.uuid),
                    'content': capsule.content,
                    'character': capsule.character,
                    'persona': capsule.persona,
                    'kind': capsule.kind.value if hasattr(capsule.kind, 'value') else str(capsule.kind),
                    'perspective': capsule.perspective,
                    'certainty': capsule.certainty,
                    'gravity': capsule.gravity,
                    'orbit_radius': capsule.orbit_radius,
                    'temporal_order': capsule.temporal_order,
                    'success_status': capsule.success_status,
                    'proven_by': capsule.proven_by,
                    'insight_potential': capsule.insight_potential,
                    'locked': capsule.locked,
                    'embedding': capsule.embedding.tolist() if hasattr(capsule, 'embedding') and capsule.embedding is not None else None,
                    'links': [str(link.uuid) for link in capsule.links] if hasattr(capsule, 'links') else [],
                    'confidence_history': [
                        [entry[0], entry[1]]  # entry[0] is already a timestamp int/float
                        for entry in capsule.confidence_history
                    ] if hasattr(capsule, 'confidence_history') else [],
                    'paradigm_shifts': capsule.paradigm_shifts if hasattr(capsule, 'paradigm_shifts') else [],
                    'intellectual_conflicts': capsule.intellectual_conflicts if hasattr(capsule, 'intellectual_conflicts') else [],
                    'pose': capsule.pose if hasattr(capsule, 'pose') else {},
                    'semantic_vector': capsule.semantic_vector.tolist() if hasattr(capsule, 'semantic_vector') and capsule.semantic_vector is not None else None,
                    'universal_id': capsule.universal_id if hasattr(capsule, 'universal_id') else None,
                    'category': capsule.category if hasattr(capsule, 'category') else None,
                    'symbol': capsule.symbol if hasattr(capsule, 'symbol') else None,
                    'language_origin': capsule.language_origin if hasattr(capsule, 'language_origin') else None,
                    'original_word': capsule.original_word if hasattr(capsule, 'original_word') else None,
                    'universal_vector': capsule.universal_vector.tolist() if hasattr(capsule, 'universal_vector') and capsule.universal_vector is not None else None,
                    'concept_categories': capsule.concept_categories if hasattr(capsule, 'concept_categories') else None,
                    'concept_count': capsule.concept_count if hasattr(capsule, 'concept_count') else None,
                    'frequency': capsule.frequency if hasattr(capsule, 'frequency') else None
                }
                capsules_data.append(capsule_dict)

            # Prepare personality data
            personality_data = {
                'cayde_personality': self.cayde_personality,
                'personality_evolution': self.personality_evolution if hasattr(self, 'personality_evolution') else [],
                'current_focus': self.current_focus if hasattr(self, 'current_focus') else 'geometric_unification'
            }

            # Combine all data
            save_data = {
                'capsules': capsules_data,
                'personality': personality_data,
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'total_capsules': len(capsules_data),
                    'version': '1.0'
                }
            }

            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Successfully saved {len(capsules_data)} capsules to {filepath}")
            return True

        except Exception as e:
            print(f"❌ Failed to save capsules to JSON: {e}")
            return False

    def load_from_json(self, filepath: str) -> bool:
        """Load capsules from a JSON file"""
        try:
            import json
            from datetime import datetime

            # Load data from file
            with open(filepath, 'r', encoding='utf-8') as f:
                save_data = json.load(f)

            capsules_data = save_data.get('capsules', [])
            personality_data = save_data.get('personality', {})

            # Clear existing capsules
            self.capsules.clear()

            # Reconstruct capsules
            uuid_map = {}  # Map string UUIDs to actual capsule objects
            for capsule_dict in capsules_data:
                # Reconstruct capsule
                capsule = Capsule(
                    content=capsule_dict['content'],
                    character=capsule_dict.get('character'),
                    persona=capsule_dict.get('persona'),
                    kind=CapsuleKind(capsule_dict['kind']) if capsule_dict['kind'] in [k.value for k in CapsuleKind] else CapsuleKind.CONCEPT,
                    perspective=capsule_dict.get('perspective'),
                    certainty=capsule_dict.get('certainty', 0.5),
                    gravity=capsule_dict.get('gravity', 0.5),
                    orbit_radius=capsule_dict.get('orbit_radius', 1.0),
                    temporal_order=capsule_dict.get('temporal_order', 0),
                    success_status=capsule_dict.get('success_status'),
                    proven_by=capsule_dict.get('proven_by'),
                    insight_potential=capsule_dict.get('insight_potential', 0.0),
                    locked=capsule_dict.get('locked', False)
                )

                # Restore embedding
                if capsule_dict.get('embedding'):
                    capsule.embedding = np.array(capsule_dict['embedding'])

                # Restore confidence history
                if capsule_dict.get('confidence_history'):
                    capsule.confidence_history = [
                        (int(entry[0]), float(entry[1]))  # Convert to (timestamp, confidence) tuple
                        for entry in capsule_dict['confidence_history']
                    ]

                # Restore other attributes
                if capsule_dict.get('paradigm_shifts'):
                    capsule.paradigm_shifts = capsule_dict['paradigm_shifts']
                if capsule_dict.get('intellectual_conflicts'):
                    capsule.intellectual_conflicts = capsule_dict['intellectual_conflicts']
                if capsule_dict.get('pose'):
                    capsule.pose = capsule_dict['pose']

                # Restore semantic/language-agnostic attributes
                if capsule_dict.get('semantic_vector'):
                    capsule.semantic_vector = np.array(capsule_dict['semantic_vector'])
                if capsule_dict.get('universal_id'):
                    capsule.universal_id = capsule_dict['universal_id']
                if capsule_dict.get('category'):
                    capsule.category = capsule_dict['category']
                if capsule_dict.get('symbol'):
                    capsule.symbol = capsule_dict['symbol']
                if capsule_dict.get('language_origin'):
                    capsule.language_origin = capsule_dict['language_origin']
                if capsule_dict.get('original_word'):
                    capsule.original_word = capsule_dict['original_word']
                if capsule_dict.get('universal_vector'):
                    capsule.universal_vector = np.array(capsule_dict['universal_vector'])
                if capsule_dict.get('concept_categories'):
                    capsule.concept_categories = capsule_dict['concept_categories']
                if capsule_dict.get('concept_count'):
                    capsule.concept_count = capsule_dict['concept_count']
                if capsule_dict.get('frequency'):
                    capsule.frequency = capsule_dict['frequency']

                # Store in UUID map and add to capsules
                uuid_map[capsule_dict['uuid']] = capsule
                self.capsules.append(capsule)

            # Restore links between capsules
            for capsule_dict in capsules_data:
                if capsule_dict.get('links'):
                    capsule = uuid_map.get(capsule_dict['uuid'])
                    if capsule:
                        capsule.links = [uuid_map[link_uuid] for link_uuid in capsule_dict['links'] if link_uuid in uuid_map]

            # Restore personality data
            if personality_data.get('cayde_personality'):
                self.cayde_personality = personality_data['cayde_personality']
            if personality_data.get('personality_evolution'):
                self.personality_evolution = personality_data['personality_evolution']
            if personality_data.get('current_focus'):
                self.current_focus = personality_data['current_focus']

            print(f"📂 Successfully loaded {len(self.capsules)} capsules from {filepath}")
            return True

        except Exception as e:
            print(f"❌ Failed to load capsules from JSON: {e}")
            return False


class RocaOrbitalMemory:
    """
    Standalone ROCA Orbital Memory System with Visualization

    Provides a complete knowledge management system with orbital visualization.
    """

    def __init__(self, width: int = 1200, height: int = 800):
        # Initialize pygame if available
        if PYGAME_AVAILABLE and not pygame.get_init():
            pygame.init()


        # Chat panel config
        self.chat_panel_width = 320
        self.width = width
        self.height = height

        if PYGAME_AVAILABLE:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("ROCA Orbital Memory System")
        else:
            self.screen = None
            print("⚠️ Pygame not available - visualization disabled")

        # Colors
        self.bg_color = (20, 20, 30)
        self.panel_color = (30, 30, 40)
        self.border_color = (60, 60, 80)
        self.highlight_color = (100, 200, 255)
        self.text_color = (255, 255, 255)

        # Fonts
        if PYGAME_AVAILABLE:
            self.font_small = pygame.font.SysFont('Arial', 12)
            self.font_medium = pygame.font.SysFont('Arial', 16)
            self.font_large = pygame.font.SysFont('Arial', 24)
        else:
            self.font_small = None
            self.font_medium = None
            self.font_large = None

        # Core memory system
        self.memory = RocaMemory()

        # Chat dialogue state (stub)
        self.chat_history = [
            ("system", "Welcome to ROCA Orbital Chat!"),
            ("user", "Hi ROCA!"),
            ("system", "How can I help you today?")
        ]
        self.chat_input = ""
        self.chat_active = False

        # Voice capabilities
        self.voice_listening = False
        self.voice_enabled = VOICE_AVAILABLE and TTS_AVAILABLE
        self.personality_profile = {
            'avg_sentence_length': 15,
            'vocabulary_complexity': 0.5,
            'formality_level': 0.5,
            'enthusiasm_level': 0.5,
            'speech_patterns': [],
            'favorite_words': Counter(),
            'communication_style': 'neutral'
        }
        
        if self.voice_enabled:
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            # Configure TTS voice to be more natural
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to set a more natural voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            self.tts_engine.setProperty('rate', 180)  # Slightly slower for clarity
            self.tts_engine.setProperty('volume', 0.8)
        else:
            self.recognizer = None
            self.tts_engine = None
            if not VOICE_AVAILABLE:
                print("⚠️ Speech recognition not available - voice input disabled")
            if not TTS_AVAILABLE:
                print("⚠️ Text-to-speech not available - voice output disabled")

        # Initialize with mathematical and physical priors
        self.memory.initialize_math_physics_priors()

        # Initialize interpolation engine for transitions
        try:
            from core.interpolation_engine import InterpolationEngine, InterpolationCurve, LayerTransform
            self.interpolation_engine = InterpolationEngine()
            self.interpolation_available = True
            print("🎞️ Interpolation engine initialized")
        except ImportError:
            self.interpolation_engine = None
            self.interpolation_available = False
            print("⚠️ Interpolation engine not available")

        # Visualization state
        """Initialize sample capsules for demonstration"""
        sample_capsules = [
            {"content": "Newton's Laws of Motion", "character": "Newton", "kind": CapsuleKind.THEORY},
            {"content": "Einstein's Theory of Relativity", "character": "Einstein", "kind": CapsuleKind.THEORY},
            {"content": "Quantum Mechanics Principles", "character": "Bohr", "kind": CapsuleKind.THEORY},
            {"content": "Thermodynamics Laws", "character": "Carnot", "kind": CapsuleKind.THEORY},
            {"content": "Calculus Fundamentals", "character": "Leibniz", "kind": CapsuleKind.METHOD},
            {"content": "Probability Theory", "character": "Bayes", "kind": CapsuleKind.METHOD},
            {"content": "Scientific Method", "character": "Galileo", "kind": CapsuleKind.METHOD},
            {"content": "Evolution Theory", "character": "Darwin", "kind": CapsuleKind.THEORY},
            {"content": "Programming Logic", "character": "Turing", "kind": CapsuleKind.METHOD},
            {"content": "Neural Networks", "character": "McCulloch", "kind": CapsuleKind.METHOD},
            {"content": "Machine Learning", "character": "Minsky", "kind": CapsuleKind.METHOD},
            {"content": "Knowledge Representation", "character": "Minsky", "kind": CapsuleKind.CONCEPT},
        ]

        for sample in sample_capsules:
            capsule = self.memory.add_capsule(
                content=sample["content"],
                perspective="scientific",
                character=sample["character"],
                kind=sample["kind"]
            )
            # Set some variety in gravity and orbit radius
            capsule.gravity = 0.3 + random.random() * 0.7
            capsule.orbit_radius = 0.5 + random.random() * 1.5

        # Visualization state
        self.show_info = False
        self.paused = False
        self.selected_capsule = None
        self.hovered_capsule = None
        self.zoom_level = 1.0
        self.camera_x = 0
        self.camera_y = 0
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.show_relationships = True
        self.show_animations = True
        self.capsule_positions = {}  # Cache for capsule positions

        if PYGAME_AVAILABLE:
            self.clock = pygame.time.Clock()
        else:
            self.clock = None

    def add_capsule(self, content: str, character: Optional[str] = None,
                   kind: CapsuleKind = CapsuleKind.CONCEPT, perspective: str = "user",
                   success_status: Optional[str] = None, proven_by: Optional[str] = None) -> Capsule:
        """Add a new capsule to the memory system"""
        return self.memory.add_capsule(content, character=character, kind=kind, perspective=perspective,
                                      success_status=success_status, proven_by=proven_by)

    def add_semantic_capsule(self, text: str, source: str = "semantic") -> List[Capsule]:
        """Create semantic capsules from text input using NLP processing"""
        # Tokenize and clean text
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if len(w) > 2 and w not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'has', 'let', 'put', 'say', 'she', 'too', 'use'}]  # Basic stop words
        
        # Create semantic capsules for key concepts
        capsules = []
        word_freq = Counter(words)
        
        # Extract top concepts
        top_concepts = word_freq.most_common(5)
        
        for word, freq in top_concepts:
            # Create semantic embedding (simple TF-IDF like)
            semantic_vector = np.random.rand(128) * freq / len(words)  # Placeholder semantic vector
            
            capsule = self.memory.add_capsule(
                content=f"Semantic: {word}",
                character=source,
                kind=CapsuleKind.CONCEPT,
                perspective="semantic"
            )
            # Attach semantic properties
            capsule.semantic_vector = semantic_vector
            capsule.frequency = freq
            capsules.append(capsule)
        
        # Create a summary capsule
        if words:
            summary_content = f"Semantic Summary: {' '.join(words[:10])}..."
            summary_capsule = self.memory.add_capsule(
                content=summary_content,
                character=source,
                kind=CapsuleKind.CONCEPT,
                perspective="semantic"
            )
            capsules.append(summary_capsule)
        
        return capsules

    def create_language_agnostic_representation(self, text: str, language: str = "en") -> List[Capsule]:
        """Create language-agnostic representations using universal concept mapping"""
        # Universal concept mapping (simplified ontology)
        universal_concepts = {
            # Basic concepts that transcend languages
            "time": {"id": "TEMPORAL_001", "category": "temporal", "universal_symbol": "⏰"},
            "space": {"id": "SPATIAL_001", "category": "spatial", "universal_symbol": "🌌"},
            "motion": {"id": "MOTION_001", "category": "physical", "universal_symbol": "🏃"},
            "energy": {"id": "ENERGY_001", "category": "physical", "universal_symbol": "⚡"},
            "life": {"id": "LIFE_001", "category": "biological", "universal_symbol": "🌱"},
            "mind": {"id": "COGNITION_001", "category": "cognitive", "universal_symbol": "🧠"},
            "communication": {"id": "COMMUNICATION_001", "category": "social", "universal_symbol": "💬"},
            "learning": {"id": "LEARNING_001", "category": "cognitive", "universal_symbol": "📚"},
            "creation": {"id": "CREATION_001", "category": "creative", "universal_symbol": "🎨"},
            "system": {"id": "SYSTEM_001", "category": "organizational", "universal_symbol": "⚙️"},
            
            # Language-specific to universal mapping
            "time|zeit|tempo|tiempo": "time",
            "space|raum|spazio|espacio": "space", 
            "motion|bewegung|moto|movimiento": "motion",
            "energy|energie|energia|energia": "energy",
            "life|leben|vita|vida": "life",
            "mind|geist|mente|mente": "mind",
            "communication|kommunikation|comunicazione|comunicacion": "communication",
            "learning|lernen|apprendimento|aprendizaje": "learning",
            "creation|schopfung|creazione|creacion": "creation",
            "system|system|sistema|sistema": "system"
        }
        
        # Tokenize and normalize
        words = re.findall(r'\b\w+\b', text.lower())
        capsules = []
        
        # Map words to universal concepts
        for word in words:
            universal_concept = None
            
            # Check direct mappings
            for key, concept in universal_concepts.items():
                if "|" in key:  # Multi-language mapping
                    if word in key.split("|"):
                        universal_concept = concept
                        break
                elif word == key:
                    universal_concept = concept
                    break
            
            if universal_concept:
                # Create language-agnostic capsule
                capsule = self.memory.add_capsule(
                    content=f"Universal: {universal_concept['universal_symbol']} {universal_concept['id']}",
                    character="universal",
                    kind=CapsuleKind.CONCEPT,
                    perspective="language_agnostic"
                )
                
                # Attach universal properties
                capsule.universal_id = universal_concept['id']
                capsule.category = universal_concept['category']
                capsule.symbol = universal_concept['universal_symbol']
                capsule.language_origin = language
                capsule.original_word = word
                
                # Create universal embedding (concept-based)
                capsule.universal_vector = np.random.rand(256)  # Higher dimensional for universal concepts
                
                capsules.append(capsule)
        
        # Create a universal representation capsule for the entire text
        if capsules:
            categories = list(set(c.category for c in capsules))
            symbols = [c.symbol for c in capsules][:5]  # Top 5 concepts
            
            universal_summary = self.memory.add_capsule(
                content=f"Universal Rep: {''.join(symbols)} ({language})",
                character="universal",
                kind=CapsuleKind.CONCEPT,
                perspective="language_agnostic"
            )
            
            universal_summary.language = language
            universal_summary.concept_categories = categories
            universal_summary.concept_count = len(capsules)
            capsules.append(universal_summary)
        
        return capsules

    def detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        text_lower = text.lower()
        
        # Language detection patterns
        if any(word in text_lower for word in ['the', 'and', 'is', 'it', 'this', 'that']):
            return 'en'
        elif any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist', 'es']):
            return 'de'
        elif any(word in text_lower for word in ['il', 'la', 'le', 'e', 'è', 'sono']):
            return 'it'
        elif any(word in text_lower for word in ['el', 'la', 'los', 'las', 'y', 'es']):
            return 'es'
        else:
            return 'en'  # Default to English

    def process_multilingual_input(self, text: str) -> List[Capsule]:
        """Process input in any supported language and create universal representations"""
        detected_lang = self.detect_language(text)
        
        # Create both semantic and universal representations
        semantic_capsules = self.add_semantic_capsule(text, source=f"multilingual_{detected_lang}")
        universal_capsules = self.create_language_agnostic_representation(text, language=detected_lang)
        
        return semantic_capsules + universal_capsules

    def update_memory(self):
        """Update memory dynamics (orbit, merge, etc.)"""
        self.memory.orbit_update()
        self.memory.apply_influences()
        self.memory.merge_similar(threshold=0.7)  # Lower threshold for more merging
        self.memory.split_conflicting()

        # Cayde generates hypotheses periodically
        if len(self.memory.capsules) > 5 and random.random() < 0.3:  # 30% chance when enough capsules
            self.memory.generate_hypotheses(1)

    def start_voice_listening(self):
        """Start voice listening mode"""
        if not self.voice_enabled:
            print("Voice capabilities not available")
            return
        
        self.voice_listening = True
        print("🎤 Voice listening activated - speak now...")
        
        # Start listening in a separate thread to avoid blocking
        import threading
        voice_thread = threading.Thread(target=self._voice_listening_loop)
        voice_thread.daemon = True
        voice_thread.start()

    def stop_voice_listening(self):
        """Stop voice listening mode"""
        self.voice_listening = False
        print("🎤 Voice listening deactivated")

    def _voice_listening_loop(self):
        """Main voice listening loop"""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                while self.voice_listening and self.running:
                    try:
                        print("🎤 Listening...")
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        
                        print("🎤 Processing speech...")
                        text = self.recognizer.recognize_google(audio)
                        
                        if text.strip():
                            print(f"🎤 Heard: '{text}'")
                            self._process_voice_input(text)
                        
                    except sr.WaitTimeoutError:
                        continue  # Continue listening
                    except sr.UnknownValueError:
                        print("🎤 Could not understand audio")
                        continue
                    except sr.RequestError as e:
                        print(f"🎤 Speech recognition error: {e}")
                        break
                        
        except Exception as e:
            print(f"🎤 Voice listening error: {e}")
            self.voice_listening = False

    def _process_voice_input(self, text: str):
        """Process voice input and update personality profile"""
        # Update personality profile based on speech patterns
        self._analyze_speech_patterns(text)
        
        # Process the input like chat input
        self.chat_history.append(("voice", text))
        
        # Create capsules from voice input
        all_capsules = self.process_multilingual_input(text)
        semantic_count = sum(1 for c in all_capsules if getattr(c, 'perspective', '') == 'semantic')
        universal_count = sum(1 for c in all_capsules if getattr(c, 'perspective', '') == 'language_agnostic')
        
        # Generate personality-mimicking response
        response = self._generate_personality_response(text)
        self.chat_history.append(("system", response))
        
        # Speak the response
        self.speak_response(response)
        
        print(f"Created {semantic_count} semantic + {universal_count} universal capsules from voice input")

    def _analyze_speech_patterns(self, text: str):
        """Analyze speech patterns to build personality profile"""
        words = text.split()
        sentences = text.split('.')
        
        # Update profile
        self.personality_profile['avg_sentence_length'] = (
            self.personality_profile['avg_sentence_length'] + len(words) / max(1, len(sentences))
        ) / 2
        
        # Vocabulary complexity (unique words / total words)
        unique_words = len(set(words))
        self.personality_profile['vocabulary_complexity'] = (
            self.personality_profile['vocabulary_complexity'] + unique_words / max(1, len(words))
        ) / 2
        
        # Favorite words
        for word in words:
            if len(word) > 3:  # Skip short words
                self.personality_profile['favorite_words'][word.lower()] += 1
        
        # Speech patterns
        if '?' in text:
            self.personality_profile['speech_patterns'].append('questioning')
        if '!' in text:
            self.personality_profile['speech_patterns'].append('enthusiastic')
        if len(sentences) > 3:
            self.personality_profile['speech_patterns'].append('verbose')
        
        # Communication style
        if len(words) < 5:
            self.personality_profile['communication_style'] = 'concise'
        elif len(sentences) > 5:
            self.personality_profile['communication_style'] = 'detailed'
        else:
            self.personality_profile['communication_style'] = 'balanced'

    def _generate_personality_response(self, user_input: str) -> str:
        """Generate a response that mimics the user's personality"""
        # Base response
        responses = [
            "That's interesting! Tell me more about that.",
            "I see what you mean. How does that connect to what we were discussing?",
            "Fascinating perspective. What led you to think that way?",
            "I understand. Can you elaborate on that point?",
            "That's a great observation. What do you think comes next?"
        ]
        
        response = random.choice(responses)
        
        # Modify based on personality profile
        profile = self.personality_profile
        
        # Adjust formality
        if profile['formality_level'] > 0.7:
            response = response.replace("That's", "That is").replace("great", "excellent")
        elif profile['formality_level'] < 0.3:
            response = response.replace("That's", "Thats").replace("great", "cool")
        
        # Adjust enthusiasm
        if profile['enthusiasm_level'] > 0.7:
            response += " I'm really excited to hear more!"
        elif profile['enthusiasm_level'] < 0.3:
            response += " Hmm, interesting."
        
        # Adjust length based on user's style
        if profile['communication_style'] == 'concise' and len(response.split()) > 10:
            # Shorten response
            response = response.split('.')[0] + '.'
        elif profile['communication_style'] == 'detailed' and len(response.split()) < 15:
            # Lengthen response
            response += " I'd love to explore this further with you."
        
        return response

    def speak_response(self, text: str):
        """Speak the response using TTS"""
        if not self.voice_enabled or not self.tts_engine:
            print(f"💬 {text}")
            return
        
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"💬 {text}")

    def toggle_voice_listening(self):
        """Toggle voice listening on/off"""
        if self.voice_listening:
            self.stop_voice_listening()
        else:
            self.start_voice_listening()

    def draw_orbital_visualization(self):
        """Draw the orbital capsule visualization and chat panel"""
        # Clear screen
        self.screen.fill(self.bg_color)

        # --- Draw chat panel (left) ---
        panel_rect = pygame.Rect(0, 0, self.chat_panel_width, self.height)
        pygame.draw.rect(self.screen, self.panel_color, panel_rect)
        pygame.draw.line(self.screen, self.border_color, (self.chat_panel_width, 0), (self.chat_panel_width, self.height), 2)

        # Chat history area
        chat_area_rect = pygame.Rect(10, 10, self.chat_panel_width - 20, self.height - 70)
        pygame.draw.rect(self.screen, (40, 40, 60), chat_area_rect)

        # Chat label
        chat_label = self.font_small.render("ROCA Chat (C=toggle chat, V=toggle voice, /mic on/off)", True, self.text_color)
        self.screen.blit(chat_label, (chat_area_rect.left + 6, chat_area_rect.top - 20))

        # Render chat messages (stub)
        y_offset = chat_area_rect.top + 8
        for sender, msg in self.chat_history[-12:]:
            color = (180, 220, 255) if sender == "user" else (255, 255, 180)
            msg_surf = self.font_small.render(f"{sender}: {msg}", True, color)
            self.screen.blit(msg_surf, (chat_area_rect.left + 6, y_offset))
            y_offset += 18

        # Chat input box
        input_rect = pygame.Rect(10, self.height - 50, self.chat_panel_width - 20, 36)
        pygame.draw.rect(self.screen, (60, 60, 80), input_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.border_color, input_rect, 2, border_radius=6)
        input_prompt = self.chat_input if self.chat_active else "Type a message..."
        input_surf = self.font_medium.render(input_prompt, True, (220, 220, 255) if self.chat_active else (150, 150, 180))
        self.screen.blit(input_surf, (input_rect.left + 8, input_rect.top + 6))

        # --- Draw orbital visualization (shifted right) ---
        vis_width = self.width - self.chat_panel_width
        vis_center_x = self.chat_panel_width + vis_width // 2
        vis_center_y = self.height // 2

        # Apply camera transform
        display_center_x = vis_center_x - self.camera_x * self.zoom_level
        display_center_y = vis_center_y - self.camera_y * self.zoom_level

        # Clear capsule positions cache
        self.capsule_positions.clear()

        # Personality core visualization (glowing circle)
        core_radius = 25
        glow_phases = [0, math.pi/4, math.pi/2, 3*math.pi/4]
        for i, phase in enumerate(glow_phases):
            glow_radius = core_radius + 8 + 4 * math.sin(time.time() * 2 + phase)
            alpha = int(30 * (1 - i/len(glow_phases)))
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.highlight_color, alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (display_center_x - glow_radius, display_center_y - glow_radius))

        # Core circle
        pygame.draw.circle(self.screen, self.highlight_color, (int(display_center_x), int(display_center_y)), core_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(display_center_x), int(display_center_y)), core_radius, 2)

        # Core label
        core_text = self.font_medium.render("ROCA Core", True, (255, 255, 220))
        core_rect = core_text.get_rect(center=(display_center_x, display_center_y + core_radius + 20))
        self.screen.blit(core_text, core_rect)

        # Draw capsules in orbital patterns
        if self.memory.capsules:
            sorted_capsules = sorted(self.memory.capsules, key=lambda c: (
                1 if c.success_status == "archived" else 0,
                -c.orbit_radius
            ))
            orbit_radii = [80, 120, 160, 200, 240, 280]
            capsules_per_orbit = [8, 12, 16, 20, 24, 28]
            capsule_index = 0
            for orbit_level, (radius, max_per_orbit) in enumerate(zip(orbit_radii, capsules_per_orbit)):
                orbit_capsules = sorted_capsules[capsule_index:capsule_index + max_per_orbit]
                capsule_index += max_per_orbit
                if not orbit_capsules:
                    continue
                for i, capsule in enumerate(orbit_capsules):
                    # Calculate orbital position with animation
                    base_angle = (i / len(orbit_capsules)) * 2 * math.pi
                    if self.show_animations:
                        angle = base_angle + time.time() * 0.1 * (orbit_level + 1) * 0.1
                    else:
                        angle = base_angle

                    # Apply zoom and camera transform
                    world_x = display_center_x + math.cos(angle) * radius * self.zoom_level
                    world_y = display_center_y + math.sin(angle) * radius * self.zoom_level

                    # Cache position for interaction
                    self.capsule_positions[capsule.uuid] = (world_x, world_y)

                    # Get color based on capsule properties
                    capsule_color = self._get_capsule_color(capsule)

                    # Draw glow effect for special capsules
                    if (self.selected_capsule == capsule or
                        (hasattr(capsule, 'insight_potential') and capsule.insight_potential > 0.7) or
                        capsule.character == "cayde"):

                        glow_radius = 20 * self.zoom_level
                        glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                        glow_color = (*capsule_color, 50)
                        pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
                        self.screen.blit(glow_surf, (world_x - glow_radius, world_y - glow_radius))

                    # Draw capsule
                    capsule_radius = int(12 * self.zoom_level)
                    pygame.draw.circle(self.screen, capsule_color, (int(world_x), int(world_y)), capsule_radius)

                    # Draw outline
                    outline_color = (255, 255, 255) if self.selected_capsule != capsule else (255, 255, 0)
                    pygame.draw.circle(self.screen, outline_color, (int(world_x), int(world_y)), capsule_radius, 2)

                    # Draw thicker outline for hovered capsule
                    if self.hovered_capsule == capsule:
                        pygame.draw.circle(self.screen, (255, 255, 255), (int(world_x), int(world_y)), capsule_radius, 3)

                    # Draw capsule label (only if zoomed in enough)
                    if self.zoom_level > 0.8:
                        name = getattr(capsule, 'character', 'Capsule') or 'Capsule'
                        # Use universal symbol for language-agnostic capsules
                        if hasattr(capsule, 'perspective') and capsule.perspective == "language_agnostic" and hasattr(capsule, 'symbol'):
                            name = capsule.symbol
                        name = str(name)[:10]
                        if len(name) > 7:
                            name = name[:7] + "..."

                        name_surf = self.font_small.render(name, True, self.text_color)
                        name_rect = name_surf.get_rect(center=(int(world_x), int(world_y + capsule_radius + 15)))
                        self.screen.blit(name_surf, name_rect)

                    # Draw relationship lines with animation
                    if self.show_relationships and hasattr(capsule, 'links') and capsule.links:
                        for link in capsule.links[:3]:
                            if link.uuid in self.capsule_positions:
                                link_pos = self.capsule_positions[link.uuid]
                                # Animated line with pulsing effect
                                if self.show_animations:
                                    pulse = (math.sin(time.time() * 3) + 1) / 2  # 0 to 1
                                    line_color = (
                                        int(100 + pulse * 100),
                                        int(100 + pulse * 50),
                                        int(150 + pulse * 105)
                                    )
                                else:
                                    line_color = (100, 100, 150)

                                pygame.draw.line(self.screen, line_color,
                                               (world_x, world_y), link_pos, 2)

            # Draw orbit rings with zoom
            for radius in orbit_radii:
                scaled_radius = radius * self.zoom_level
                if scaled_radius > 20:  # Only draw if visible
                    pygame.draw.circle(self.screen, (80, 80, 120),
                                     (int(display_center_x), int(display_center_y)),
                                     int(scaled_radius), 1)
        # End orbital visualization

        # Draw waveform widgets for audio capsules
        self.draw_waveform_widgets(self.screen)

        # Draw information panel
        if self.show_info:
            self._draw_info_panel()

    def _handle_capsule_click(self, mouse_x: int, mouse_y: int):
        """Handle mouse click on capsules"""
        center_x = self.chat_panel_width + (self.width - self.chat_panel_width) // 2
        center_y = self.height // 2

        # Apply camera transform
        world_x = (mouse_x - center_x) / self.zoom_level + self.camera_x + center_x
        world_y = (mouse_y - center_y) / self.zoom_level + self.camera_y + center_y

        clicked_capsule = None
        min_distance = float('inf')

        for capsule_uuid, pos in self.capsule_positions.items():
            capsule = next((c for c in self.memory.capsules if c.uuid == capsule_uuid), None)
            if capsule is None:
                continue
            dx = world_x - pos[0]
            dy = world_y - pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < 20 / self.zoom_level and distance < min_distance:  # 20 pixel click radius
                clicked_capsule = capsule
                min_distance = distance

        if clicked_capsule:
            if self.selected_capsule == clicked_capsule:
                # Double-click effect - focus on this capsule
                self._focus_on_capsule(clicked_capsule)
            else:
                self.selected_capsule = clicked_capsule
                print(f"Selected capsule: {clicked_capsule.content[:50]}...")
        else:
            self.selected_capsule = None

    def _handle_capsule_hover(self, mouse_x: int, mouse_y: int):
        """Handle mouse hover over capsules"""
        center_x = self.chat_panel_width + (self.width - self.chat_panel_width) // 2
        center_y = self.height // 2

        # Apply camera transform
        world_x = (mouse_x - center_x) / self.zoom_level + self.camera_x + center_x
        world_y = (mouse_y - center_y) / self.zoom_level + self.camera_y + center_y

        hovered_capsule = None
        min_distance = float('inf')

        for capsule_uuid, pos in self.capsule_positions.items():
            capsule = next((c for c in self.memory.capsules if c.uuid == capsule_uuid), None)
            if capsule is None:
                continue
            dx = world_x - pos[0]
            dy = world_y - pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < 25 / self.zoom_level and distance < min_distance:  # Slightly larger hover radius
                hovered_capsule = capsule
                min_distance = distance

        self.hovered_capsule = hovered_capsule

    def _focus_on_capsule(self, capsule):
        """Focus the view on a specific capsule"""
        if capsule.uuid in self.capsule_positions:
            pos = self.capsule_positions[capsule.uuid]
            center_x = self.chat_panel_width + (self.width - self.chat_panel_width) // 2
            center_y = self.height // 2

            # Center camera on the capsule
            self.camera_x = pos[0] - center_x
            self.camera_y = pos[1] - center_y
            self.zoom_level = 2.0  # Zoom in for focus

    def _get_capsule_color(self, capsule) -> Tuple[int, int, int]:
        """Get color for capsule based on type, confidence, and state"""
        # Base colors by capsule kind
        kind_colors = {
            CapsuleKind.FACT: (100, 150, 255),      # Blue
            CapsuleKind.CONCEPT: (150, 255, 150),   # Green
            CapsuleKind.EVENT: (255, 200, 100),     # Orange
            CapsuleKind.PERSON: (255, 150, 200),    # Pink
            CapsuleKind.THEORY: (200, 150, 255),    # Purple
            CapsuleKind.METHOD: (150, 200, 255),    # Light blue
            CapsuleKind.OBSERVATION: (255, 255, 150), # Yellow
            CapsuleKind.HYPOTHESIS: (255, 150, 150), # Red
            CapsuleKind.AUDIO: (100, 255, 200),     # Teal
        }

        base_color = kind_colors.get(capsule.kind, (150, 150, 150))

        # Modify based on confidence/certainty
        if hasattr(capsule, 'pose') and 'certainty' in capsule.pose:
            certainty = capsule.pose['certainty']
            # Blend with confidence color
            conf_color = (int(base_color[0] * certainty + 255 * (1-certainty)),
                         int(base_color[1] * certainty + 100 * (1-certainty)),
                         int(base_color[2] * certainty + 100 * (1-certainty)))
            base_color = conf_color

        # Special colors for specific states
        if hasattr(capsule, 'perspective'):
            if capsule.perspective == "language_agnostic":
                base_color = (200, 0, 200)  # Bright purple
            elif capsule.perspective == "semantic":
                base_color = (0, 255, 100)  # Bright green

        if capsule.character == "cayde":
            base_color = (0, 255, 255)  # Cyan

        if hasattr(capsule, 'success_status'):
            if capsule.success_status == "archived":
                base_color = (80, 60, 60)  # Dark gray
            elif capsule.success_status == "proven_later":
                base_color = (255, 0, 255)  # Magenta

        if hasattr(capsule, 'insight_potential') and capsule.insight_potential > 0.7:
            base_color = (255, 255, 0)  # Bright yellow

        # Highlight selected capsule
        if self.selected_capsule == capsule:
            # Brighten the color
            base_color = (min(255, base_color[0] + 50),
                         min(255, base_color[1] + 50),
                         min(255, base_color[2] + 50))

        # Glow effect for hovered capsule
        if self.hovered_capsule == capsule:
            base_color = (min(255, base_color[0] + 30),
                         min(255, base_color[1] + 30),
                         min(255, base_color[2] + 30))

        return base_color
        """Draw information panel with memory statistics"""
        panel_width = 300
        panel_height = 200
        panel_x = self.width - panel_width - 10
        panel_y = 10

        # Panel background
        pygame.draw.rect(self.screen, self.panel_color, (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.border_color, (panel_x, panel_y, panel_width, panel_height), 2)

        # Title
        title = self.font_medium.render("ROCA Memory Status", True, self.highlight_color)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))

        # Statistics
        cayde_capsules = [c for c in self.memory.capsules if c.character == "cayde"]
        insights = [c for c in self.memory.capsules if hasattr(c, 'insight_potential') and c.insight_potential > 0.7]
        proven_later = [c for c in self.memory.capsules if hasattr(c, 'success_status') and c.success_status == "proven_later"]

        # GPU memory info
        gpu_info = self.memory.gpu_memory_info()
        gpu_mem_str = ""
        if gpu_info["gpu_available"]:
            mem_info = gpu_info["memory_info"]
            if gpu_info["gpu_type"] == "NVIDIA" and "used_bytes" in mem_info:
                used_mb = mem_info["used_bytes"] / (1024 * 1024)
                total_mb = mem_info["total_bytes"] / (1024 * 1024)
                gpu_mem_str = f"GPU: {used_mb:.1f}/{total_mb:.1f}MB"
            elif gpu_info["gpu_type"] == "AMD" and "allocated_bytes" in mem_info:
                allocated_mb = mem_info["allocated_bytes"] / (1024 * 1024)
                gpu_mem_str = f"GPU: {allocated_mb:.1f}MB used"

        stats = [
            f"Capsules: {len(self.memory.capsules)}",
            f"Core Knowledge: {len(self.memory.get_core_capsules())}",
            f"Cayde Capsules: {len(cayde_capsules)}",
            f"Insights Detected: {len(insights)}",
            f"Theories Proven Later: {len(proven_later)}",
            f"GPU: {GPU_TYPE} {'✓' if GPU_AVAILABLE else '✗'}" + (f" | {gpu_mem_str}" if gpu_mem_str else ""),
            f"Voice: {'🎤 ON' if self.voice_listening else '🔇 OFF'} {'✓' if self.voice_enabled else '✗'}",
            "",
            "Controls:",
            "SPACE: Update memory",
            "S: Save memory to JSON",
            "L: Load memory from JSON",
            "G: Optimize GPU memory",
            "C: Toggle chat",
            "V: Toggle voice",
            "I: Toggle info",
            "R: Toggle relationships",
            "A: Toggle animations",
            "Mouse: Click capsules",
            "Wheel: Zoom in/out",
            "Right-click: Pan view",
            "",
            "Chat Commands:",
            "/mic on: Enable microphone",
            "/mic off: Disable microphone",
            "ESC: Exit"
        ]

        y_offset = panel_y + 40
        for stat in stats:
            color = self.text_color if not stat.startswith("GPU") else (100, 255, 100) if GPU_AVAILABLE else (255, 100, 100)
            text = self.font_small.render(stat, True, color)
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 20

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.update_memory()
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_r:  # Toggle relationship lines
                    self.show_relationships = not self.show_relationships
                    print(f"Relationship lines: {'ON' if self.show_relationships else 'OFF'}")
                elif event.key == pygame.K_a:  # Toggle animations
                    self.show_animations = not self.show_animations
                    print(f"Animations: {'ON' if self.show_animations else 'OFF'}")
                elif event.key == pygame.K_v:  # Toggle voice listening
                    self.toggle_voice_listening()
                elif event.key == pygame.K_s:  # Save capsules to JSON
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    filepath = filedialog.asksaveasfilename(
                        defaultextension=".json",
                        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                        title="Save ROCA Memory"
                    )
                    if filepath:
                        success = self.memory.save_to_json(filepath)
                        if success:
                            self.chat_history.append(("system", f"💾 Memory saved to {filepath}"))
                        else:
                            self.chat_history.append(("system", "❌ Failed to save memory"))
                elif event.key == pygame.K_l:  # Load capsules from JSON
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    filepath = filedialog.askopenfilename(
                        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                        title="Load ROCA Memory"
                    )
                    if filepath:
                        success = self.memory.load_from_json(filepath)
                        if success:
                            self.chat_history.append(("system", f"📂 Memory loaded from {filepath}"))
                            # Update visualization after loading
                            self.update_memory()
                        else:
                            self.chat_history.append(("system", "❌ Failed to load memory"))
                elif event.key == pygame.K_g:  # Optimize GPU memory
                    self.memory.optimize_gpu_memory()
                    gpu_info = self.memory.gpu_memory_info()
                    if gpu_info["gpu_available"]:
                        mem_info = gpu_info["memory_info"]
                        if gpu_info["gpu_type"] == "NVIDIA" and "used_bytes" in mem_info:
                            used_mb = mem_info["used_bytes"] / (1024 * 1024)
                            self.chat_history.append(("system", f"🧹 GPU memory optimized | Used: {used_mb:.1f}MB"))
                        elif gpu_info["gpu_type"] == "AMD" and "allocated_bytes" in mem_info:
                            allocated_mb = mem_info["allocated_bytes"] / (1024 * 1024)
                            self.chat_history.append(("system", f"🧹 GPU memory optimized | Used: {allocated_mb:.1f}MB"))
                        else:
                            self.chat_history.append(("system", "🧹 GPU memory optimized"))
                    else:
                        self.chat_history.append(("system", "⚠️ No GPU available for optimization"))
                elif event.key == pygame.K_RETURN and self.chat_active:
                    # Process chat input
                    if self.chat_input.strip():
                        # Check for mic commands
                        if self.chat_input.strip().lower() == "/mic on":
                            if not self.voice_listening:
                                self.start_voice_listening()
                                self.chat_history.append(("system", "🎤 Microphone activated"))
                            else:
                                self.chat_history.append(("system", "🎤 Microphone is already active"))
                            self.chat_input = ""
                            self.chat_active = False
                        elif self.chat_input.strip().lower() == "/mic off":
                            if self.voice_listening:
                                self.stop_voice_listening()
                                self.chat_history.append(("system", "🔇 Microphone deactivated"))
                            else:
                                self.chat_history.append(("system", "🔇 Microphone is already inactive"))
                            self.chat_input = ""
                            self.chat_active = False
                        else:
                            # Regular chat processing
                            self.chat_history.append(("user", self.chat_input))
                            # Create multilingual representations from input
                            all_capsules = self.process_multilingual_input(self.chat_input)
                            semantic_count = sum(1 for c in all_capsules if getattr(c, 'perspective', '') == 'semantic')
                            universal_count = sum(1 for c in all_capsules if getattr(c, 'perspective', '') == 'language_agnostic')
                            response = f"Created {semantic_count} semantic + {universal_count} universal capsules."
                            self.chat_history.append(("system", response))
                            self.chat_input = ""
                    self.chat_active = False
                elif event.key == pygame.K_BACKSPACE and self.chat_active:
                    self.chat_input = self.chat_input[:-1]
                elif event.key == pygame.K_c:  # Toggle chat
                    self.chat_active = not self.chat_active
                    if not self.chat_active:
                        self.chat_input = ""
            elif event.type == pygame.TEXTINPUT and self.chat_active:
                self.chat_input += event.text
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self._handle_capsule_click(mouse_x, mouse_y)
                elif event.button == 3:  # Right click
                    self.dragging = True
                    self.drag_start_x, self.drag_start_y = pygame.mouse.get_pos()
                elif event.button == 4:  # Mouse wheel up - zoom in
                    self.zoom_level = min(3.0, self.zoom_level * 1.2)
                elif event.button == 5:  # Mouse wheel down - zoom out
                    self.zoom_level = max(0.3, self.zoom_level / 1.2)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:  # Right click release
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self._handle_capsule_hover(mouse_x, mouse_y)

                if self.dragging:
                    # Pan the view
                    dx = mouse_x - self.drag_start_x
                    dy = mouse_y - self.drag_start_y
                    self.camera_x -= dx / self.zoom_level
                    self.camera_y -= dy / self.zoom_level
                    self.drag_start_x, self.drag_start_y = mouse_x, mouse_y

    def visualize(self):
        """Run the orbital visualization"""
        self.running = True

        while self.running:
            self.handle_events()
            self.draw_orbital_visualization()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "total_capsules": len(self.memory.capsules),
            "core_capsules": len(self.memory.get_core_capsules()),
            "gpu_accelerated": GPU_AVAILABLE,
            "gpu_type": GPU_TYPE,
            "capsule_types": {kind.value: sum(1 for c in self.memory.capsules if c.kind == kind)
                            for kind in CapsuleKind}
        }

    def get_cayde_personality_report(self) -> str:
        """Get Cayde's personality report"""
        return self.memory.get_cayde_personality_report()

    # ===== INTERPOLATION ENGINE METHODS =====

    def create_transition_capsule(self, start_content: str, end_content: str,
                                curve: str = "ease_in_out", steps: int = 10,
                                use_shape_morphing: bool = False) -> Capsule:
        """
        Create a transition capsule with interpolation metadata

        Args:
            start_content: Content of the starting keyframe
            end_content: Content of the ending keyframe
            curve: Interpolation curve type
            steps: Number of interpolation steps
            use_shape_morphing: Whether to use shape-aware morphing

        Returns:
            Transition capsule with interpolation metadata
        """
        if not self.interpolation_available:
            # Fallback to basic transition capsule
            transition_content = f"Transition: {start_content[:30]}... → {end_content[:30]}..."
            return self.memory.add_capsule(transition_content, kind=CapsuleKind.TRANSITION,
                                         perspective="interpolation")

        # Create interpolation metadata
        from core.interpolation_engine import LayerTransform

        start_transform = LayerTransform()  # Default transform
        end_transform = LayerTransform(position=(100, 50), scale=(1.5, 1.5), rotation=45.0, opacity=0.8)

        metadata = self.interpolation_engine.create_interpolation_metadata(
            start_transform, end_transform, curve, steps, use_shape_morphing
        )

        # Create transition capsule with metadata embedded in content
        transition_content = f"🎞️ Bezier Transition ({curve}, {steps} steps): {start_content[:25]}... → {end_content[:25]}..."
        capsule = self.memory.add_capsule(transition_content, kind=CapsuleKind.METHOD,
                                        perspective="bezier_interpolation")

        # Store metadata in capsule pose for later retrieval
        capsule.pose["interpolation_metadata"] = metadata

        return capsule

    def generate_interpolation_frames(self, start_image: np.ndarray, end_image: np.ndarray,
                                    curve: str = "ease_in_out", steps: int = 10,
                                    use_shape_morphing: bool = False) -> List[np.ndarray]:
        """
        Generate interpolated frames between two images using the interpolation engine

        Args:
            start_image: Starting image/frame
            end_image: Ending image/frame
            curve: Interpolation curve to use
            steps: Number of interpolation steps
            use_shape_morphing: Whether to use shape-aware morphing

        Returns:
            List of interpolated images
        """
        if not self.interpolation_available:
            print("⚠️ Interpolation engine not available")
            return []

        try:
            return self.interpolation_engine.bezier_interpolate(
                start_image, end_image, steps, curve, use_shape_morphing
            )
        except Exception as e:
            print(f"⚠️ Interpolation failed: {e}")
            return []

    def get_interpolation_presets(self) -> List[str]:
        """Get available interpolation curve presets"""
        if not self.interpolation_available:
            return ["linear"]

        from core.interpolation_engine import InterpolationCurve
        return [curve.value for curve in InterpolationCurve]

    # ===== AUDIO WAVEFORM METHODS =====

    def create_audio_capsule(self, audio_path: str, content: str = "",
                           character: Optional[str] = None) -> 'AudioCapsule':
        """
        Create an audio capsule with waveform data

        Args:
            audio_path: Path to the audio file
            content: Text content describing the audio
            character: Associated character/persona

        Returns:
            AudioCapsule with loaded waveform data
        """
        try:
            from core.audio_waveform import create_audio_capsule_from_file
            audio_capsule = create_audio_capsule_from_file(audio_path, content)

            # Add to memory system
            self.memory.capsules.append(audio_capsule)

            # Set additional properties
            if character:
                audio_capsule.character = character

            print(f"🔊 Created audio capsule: {audio_path}")
            return audio_capsule

        except Exception as e:
            print(f"Failed to create audio capsule: {e}")
            # Fallback to regular capsule
            return self.memory.add_capsule(content or f"Audio: {audio_path}",
                                         kind=CapsuleKind.AUDIO, character=character)

    def get_audio_capsules(self) -> List['AudioCapsule']:
        """Get all audio capsules in the system"""
        return [c for c in self.memory.capsules
                if hasattr(c, 'kind') and c.kind == CapsuleKind.AUDIO]

    def create_waveform_widget(self, audio_capsule: 'AudioCapsule',
                              x: int, y: int, width: int) -> Optional['AudioWaveformWidget']:
        """
        Create a waveform widget for an audio capsule

        Args:
            audio_capsule: The audio capsule to visualize
            x, y: Position of the widget
            width: Width of the widget

        Returns:
            AudioWaveformWidget instance or None if failed
        """
        try:
            return audio_capsule.create_waveform_widget(x, y, width)
        except Exception as e:
            print(f"Failed to create waveform widget: {e}")
            return None

    def update_waveform_playhead(self, audio_capsule: 'AudioCapsule', time_seconds: float):
        """Update the playhead position for an audio capsule's waveform"""
        if hasattr(audio_capsule, 'waveform_widget') and audio_capsule.waveform_widget:
            audio_capsule.waveform_widget.set_playhead_time(time_seconds)

    def handle_waveform_events(self, event: pygame.event.Event) -> bool:
        """Handle pygame events for all active waveform widgets"""
        for capsule in self.get_audio_capsules():
            if (hasattr(capsule, 'waveform_widget') and
                capsule.waveform_widget and
                capsule.waveform_widget.handle_event(event)):
                return True
        return False

    def draw_waveform_widgets(self, surface: pygame.Surface):
        """Draw all active waveform widgets"""
        for capsule in self.get_audio_capsules():
            if hasattr(capsule, 'waveform_widget') and capsule.waveform_widget:
                capsule.waveform_widget.draw(surface)

    # ===== DOCUMENT PROCESSING AND USER INTERACTION METHODS =====

    def ingest_document(self, text: str, source: str = "document", author: Optional[str] = None) -> List[Capsule]:
        """Delegate to memory system"""
        return self.memory.ingest_document(text, source, author)

    def process_user_input(self, user_message: str, use_voice: bool = False) -> Dict[str, Any]:
        """Delegate to memory system"""
        return self.memory.process_user_input(user_message, use_voice)

    def generate_future_breakthroughs(self, num_hypotheses: int = 3) -> List[Capsule]:
        """Delegate to memory system"""
        return self.memory.generate_future_breakthroughs(num_hypotheses)

    def run_century_progression_learning(self):
        """Run the complete historical learning progression from Descartes/Newton through centuries"""
        print("🕰️  Beginning Cayde's Historical Learning Journey")
        print("=" * 60)

        curriculum = self.memory.create_historical_learning_curriculum()

        # Group by century for organized learning
        centuries = {
            '17th Century': [],
            '18th Century': [],
            '19th Century': [],
            '20th Century': []
        }

        for item in curriculum:
            if 'Descartes' in str(item.get('scientist', '')) or 'Newton' in str(item.get('scientist', '')) or 'Scientific_Revolution' in str(item.get('scientist', '')) or 'Leibniz' in str(item.get('scientist', '')):
                centuries['17th Century'].append(item)
            elif '18' in str(item.get('scientist', '')) or 'Enlightenment' in str(item.get('scientist', '')) or 'Euler' in str(item.get('scientist', '')) or 'Lagrange' in str(item.get('scientist', '')) or 'd\'Alembert' in str(item.get('scientist', '')):
                centuries['18th Century'].append(item)
            elif '19' in str(item.get('scientist', '')) or 'Industrial' in str(item.get('scientist', '')) or 'Gauss' in str(item.get('scientist', '')) or 'Hamilton' in str(item.get('scientist', '')) or 'Maxwell' in str(item.get('scientist', '')) or 'Boltzmann' in str(item.get('scientist', '')) or 'Cantor' in str(item.get('scientist', '')):
                centuries['19th Century'].append(item)
            else:
                centuries['20th Century'].append(item)

        # Learn through each century
        for century, learning_sequence in centuries.items():
            if learning_sequence:  # Only process centuries with content
                print(f"\n📚 {century} Learning Phase")
                print("-" * 40)

                # Experience the century's developments
                for item in learning_sequence:
                    capsule = self.memory.add_capsule(
                        content=item['content'],
                        character=item.get('scientist', century),
                        kind=item.get('kind', CapsuleKind.CONCEPT),
                        success_status=item.get('initial_status', 'unknown'),
                        perspective=item.get('perspective', 'historical')
                    )
                    print(f"  📖 Learned: {item['title'][:50]}...")

                # Allow critical analysis after each century
                if len(self.memory.cayde_personality['personality_evolution']) > 3:
                    print(f"\n🧠 Cayde reflecting on {century}...")
                    critical_insights = self.memory.cayde_critical_analysis_session()
                    if critical_insights:
                        print(f"   Generated {len(critical_insights)} critical insights from {century}")

                # Show personality development
                print(f"\n🧠 Cayde's personality after {century}:")
                evolution_count = len(self.memory.cayde_personality['personality_evolution'])
                if evolution_count > 0:
                    recent = self.memory.cayde_personality['personality_evolution'][-1]
                    print(f"   Latest evolution: {recent.get('insight_type', 'N/A')} → {recent.get('personality_change', 'N/A')}")

        # Final reflection
        print(f"\n🎯 Cayde's Complete Historical Journey Complete!")
        print(f"   Learned from {len(curriculum)} historical developments")
        print(f"   Experienced {len(self.memory.cayde_personality['personality_evolution'])} personality evolutions")
        print(f"   Generated {len(self.memory.cayde_personality['disagreements_with_scientists'])} scientific disagreements")
        print(f"   Abandoned {len(self.memory.cayde_personality['abandoned_intuitions'])} limiting intuitions")

        # Extract philosophical lessons from the historical debates
        print(f"\n🧠 Cayde reflecting on the deeper lessons of scientific history...")
        historical_lessons = self.memory.extract_historical_lessons()

        print(f"   Extracted {len(historical_lessons)} philosophical insights about scientific progress")

        return curriculum

    def cognitive_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Delegate cognitive queries to the memory system's cognitive layer"""
        return self.memory.cognitive_query(query, context)


# ===== STANDALONE FUNCTIONS =====

def create_roca_memory() -> RocaOrbitalMemory:
    """Create a new ROCA orbital memory system"""
    return RocaOrbitalMemory()

def test_physics_engine():
    """Test Cayde's internal physics engine with inconsistent claims"""
    print("🧪 Testing Cayde's Internal Physics Engine")
    print("=" * 50)

    # Initialize Cayde
    cayde = RocaOrbitalMemory()

    # Add some consistent foundational knowledge
    cayde.add_capsule("Energy conservation: Energy cannot be created or destroyed, only transformed",
                     perspective="scientific_law", character="physics", kind=CapsuleKind.CONCEPT)
    cayde.add_capsule("Speed of light in vacuum: c = 299,792,458 m/s",
                     perspective="measured_constant", character="physics", kind=CapsuleKind.CONCEPT)
    cayde.add_capsule("Mass-energy equivalence: E = mc² where m is rest mass",
                     perspective="scientific_law", character="einstein", kind=CapsuleKind.CONCEPT)

    print("\n📚 Added foundational physics knowledge\n")

    # Test 1: Add a logically consistent claim
    print("Test 1: Adding logically consistent claim")
    consistent_capsule = cayde.add_capsule(
        "Photon energy: E = hc/λ where h is Planck's constant and λ is wavelength",
        perspective="derived_law", character="physics", kind=CapsuleKind.CONCEPT
    )
    print(f"✅ Consistent claim accepted (certainty: {consistent_capsule.certainty:.2f})\n")

    # Test 2: Add a contradictory claim (violates energy conservation)
    print("Test 2: Adding contradictory claim (violates energy conservation)")
    contradictory_capsule = cayde.add_capsule(
        "Perpetual motion machine: A machine that runs forever without energy input",
        perspective="hypothetical", character="inventor", kind=CapsuleKind.CONCEPT
    )
    print(f"Result: {'Accepted' if contradictory_capsule in cayde.memory.capsules else 'Rejected'}")
    if contradictory_capsule in cayde.memory.capsules:
        print(f"Certainty: {contradictory_capsule.certainty:.2f}\n")

    # Test 3: Add a dimensionally inconsistent claim
    print("Test 3: Adding dimensionally inconsistent claim")
    dimensional_capsule = cayde.add_capsule(
        "Force equals mass times velocity: F = m*v (should be F = m*a)",
        perspective="erroneous", character="student", kind=CapsuleKind.CONCEPT
    )
    print(f"Result: {'Accepted' if dimensional_capsule in cayde.memory.capsules else 'Rejected'}")
    if dimensional_capsule in cayde.memory.capsules:
        print(f"Certainty: {dimensional_capsule.certainty:.2f}\n")

    # Test 4: Add a unit inconsistent claim
    print("Test 4: Adding unit inconsistent claim")
    unit_capsule = cayde.add_capsule(
        "Light speed in mph: c = 670,616,629 mph (correctly calculated)",
        perspective="converted", character="physics", kind=CapsuleKind.CONCEPT
    )
    print(f"✅ Unit consistent claim accepted (certainty: {unit_capsule.certainty:.2f})\n")

    # Test 5: Add a logically impossible claim
    print("Test 5: Adding logically impossible claim")
    impossible_capsule = cayde.add_capsule(
        "Square circle: A geometric shape that is both a square and a circle simultaneously",
        perspective="paradox", character="philosopher", kind=CapsuleKind.CONCEPT
    )
    print(f"Result: {'Accepted' if impossible_capsule in cayde.memory.capsules else 'Rejected'}")
    if impossible_capsule in cayde.memory.capsules:
        print(f"Certainty: {impossible_capsule.certainty:.2f}\n")

    # Run critical analysis session
    print("🧠 Running critical analysis session with physics engine...")
    critical_insights = cayde.memory.cayde_critical_analysis_session()
    print(f"Generated {len(critical_insights)} critical insights that passed consistency checks\n")

    # Show final memory state
    print("📊 Final Memory State:")
    print(f"Total capsules: {len(cayde.memory.capsules)}")
    consistent_count = sum(1 for c in cayde.memory.capsules if c.certainty > 0.8)
    weakened_count = sum(1 for c in cayde.memory.capsules if 0.3 < c.certainty <= 0.8)
    rejected_count = sum(1 for c in cayde.memory.capsules if c.certainty <= 0.3)
    print(f"High confidence (certainty > 0.8): {consistent_count}")
    print(f"Weakened (0.3 < certainty ≤ 0.8): {weakened_count}")
    print(f"Rejected/low confidence (certainty ≤ 0.3): {rejected_count}")

    print("\n🎯 Physics Engine Test Complete!")
    return cayde


def test_mathematical_interpretation():
    """Test Cayde's mathematical interpretation system - hard-coded operations with philosophical shaping"""
    print("🔢 Testing Cayde's Mathematical Interpretation System")
    print("=" * 60)

    # Initialize Cayde
    cayde = RocaOrbitalMemory()

    print("📐 Cayde's Mathematical Philosophy:")
    personality = cayde.memory.cayde_personality
    print(f"   Platonism: {personality['mathematical_platonism']['level']:.2f} - Math truths are discovered")
    print(f"   Geometric Realism: {personality['geometric_realism']['level']:.2f} - Geometry is physical")
    print(f"   Mathematical Skepticism: {personality['mathematical_skepticism']['level']:.2f} - Questions necessity")
    print()

    # Test 1: Basic arithmetic operations (hard-coded)
    print("Test 1: Basic Arithmetic Operations")
    add_result = cayde.memory.perform_mathematical_operation('add', [3, 5, 7])
    print(f"   3 + 5 + 7 = {add_result['result']}")

    multiply_result = cayde.memory.perform_mathematical_operation('multiply', [4, 5])
    print(f"   4 × 5 = {multiply_result['result']}")

    power_result = cayde.memory.perform_mathematical_operation('power', [2, 8])
    print(f"   2^8 = {power_result['result']}")
    print()

    # Test 2: Calculus operations (hard-coded)
    print("Test 2: Calculus Operations")
    deriv_result = cayde.memory.perform_mathematical_operation('derivative', ['x^2'])
    print(f"   d/dx(x²) = {deriv_result['result']}")

    integral_result = cayde.memory.perform_mathematical_operation('integral', ['cos(x)'])
    print(f"   ∫cos(x)dx = {integral_result['result']}")
    print()

    # Test 3: Vector operations (hard-coded)
    print("Test 3: Vector Operations")
    dot_result = cayde.memory.perform_mathematical_operation('vector_dot_product', [[1, 2, 3], [4, 5, 6]])
    print(f"   [1,2,3] • [4,5,6] = {dot_result['result']}")

    cross_result = cayde.memory.perform_mathematical_operation('cross_product', [[1, 0, 0], [0, 1, 0]])
    print(f"   [1,0,0] × [0,1,0] = {cross_result['result']}")
    print()

    # Test 4: Apply philosophical interpretation (shapes reasoning)
    print("Test 4: Philosophical Interpretation of Results")
    philosophical_add = cayde.memory.apply_mathematical_philosophy(add_result)
    print(f"   Arithmetic result with philosophy: {philosophical_add.get('philosophical_note', 'No special interpretation')}")

    # Test mathematical expression in capsule
    print("Test 5: Mathematical Expression Validation")
    math_capsule = cayde.add_capsule(
        "E = mc² where E is energy, m is mass, c is speed of light",
        perspective="physical_law", character="einstein", kind=CapsuleKind.THEORY
    )
    print(f"   Mathematical capsule certainty: {math_capsule.certainty:.2f}")

    # Test invalid mathematical expression
    print("Test 6: Invalid Mathematical Expression")
    invalid_capsule = cayde.add_capsule(
        "Division by zero: x = 5/0",
        perspective="erroneous", character="student", kind=CapsuleKind.CONCEPT
    )
    print(f"   Invalid math capsule certainty: {invalid_capsule.certainty:.2f}")
    print()

    # Demonstrate philosophical debates shaping reasoning
    print("🧠 Philosophical Debates Shaping Mathematical Reasoning:")
    print("   • Platonism makes Cayde see E=mc² as a 'discovered eternal truth'")
    print("   • Geometric realism makes him interpret spacetime curvature as 'physical reality itself'")
    print("   • Mathematical skepticism makes him question if complex math is 'physically necessary'")
    print()

    # Show how philosophy affects confidence
    print("📊 Philosophy's Effect on Mathematical Confidence:")
    # Test with a result that triggers platonism
    platonist_result = cayde.memory.apply_mathematical_philosophy({'operation': 'add', 'result': 42})
    print(f"   Platonist interpretation: {platonist_result.get('philosophical_note', 'No note')} (confidence ×{platonist_result.get('confidence_multiplier', 1.0)})")

    # Test with a result that triggers skepticism (different operation)
    skeptic_result = cayde.memory.apply_mathematical_philosophy({'operation': 'solve_quadratic', 'result': [1, -1]})
    print(f"   Skeptic interpretation: {skeptic_result.get('skeptical_note', 'No note')} (confidence ×{skeptic_result.get('confidence_multiplier', 1.0)})")
    print()

    print("🎯 Mathematical Interpretation Test Complete!")
    print("Operations are hard-coded, but philosophical meaning shapes reasoning style!")
    return cayde


def test_time_awareness():
    """Test Cayde's time awareness system - prevent anachronistic reasoning"""
    print("⏰ Testing Cayde's Time Awareness System")
    print("=" * 50)

    # Initialize Cayde
    cayde = RocaOrbitalMemory()

    print(f"📅 Cayde's temporal context: Year {cayde.memory.cayde_personality['time_awareness']['current_year']}")
    print("🚫 Future knowledge is inaccessible\n")

    # Test 1: Add knowledge from Cayde's time period (should be accepted)
    print("Test 1: Adding contemporary knowledge (1905)")
    contemporary_capsule = cayde.add_capsule(
        "Special theory of relativity: The laws of physics are the same in all inertial frames",
        perspective="revolutionary", character="einstein", kind=CapsuleKind.THEORY
    )
    print(f"✅ Contemporary knowledge accepted (certainty: {contemporary_capsule.certainty:.2f})\n")

    # Test 2: Try to add future knowledge (should be heavily penalized)
    print("Test 2: Attempting to add future knowledge (quantum mechanics from 1920s)")
    future_capsule = cayde.add_capsule(
        "Quantum mechanics: Particles can exist in superposition states",
        perspective="future_knowledge", character="future_physicist", kind=CapsuleKind.THEORY
    )
    print(f"Result: {'Accepted' if future_capsule in cayde.memory.capsules else 'Rejected'}")
    if future_capsule in cayde.memory.capsules:
        print(f"Certainty: {future_capsule.certainty:.2f} (heavily penalized for anachronism)\n")

    # Test 3: Try to add very future knowledge (AI concepts)
    print("Test 3: Attempting to add very future knowledge (AI from 2020s)")
    ai_capsule = cayde.add_capsule(
        "Artificial intelligence: Machines can learn and make decisions like humans",
        perspective="futuristic", character="ai_researcher", kind=CapsuleKind.CONCEPT
    )
    print(f"Result: {'Accepted' if ai_capsule in cayde.memory.capsules else 'Rejected'}")
    if ai_capsule in cayde.memory.capsules:
        print(f"Certainty: {ai_capsule.certainty:.2f} (severely penalized for extreme anachronism)\n")

    # Test 4: Add novel idea relative to current time
    print("Test 4: Adding novel idea for 1905 (unified field theory quest)")
    novel_capsule = cayde.add_capsule(
        "Unified field theory: Gravity and electromagnetism should be unified",
        perspective="speculative", character="einstein", kind=CapsuleKind.THEORY
    )
    print(f"✅ Novel contemporary idea accepted (certainty: {novel_capsule.certainty:.2f})\n")

    # Test temporal novelty checking directly
    print("🧪 Direct temporal novelty tests:")
    test_ideas = [
        ("Relativity principle", "contemporary concept"),
        ("Quantum entanglement", "future concept (1935)"),
        ("General relativity", "contemporary but novel (1915)"),
        ("Blockchain technology", "far future concept (2008)"),
        ("Neural networks", "future concept (1940s)")
    ]

    for idea, description in test_ideas:
        test_capsule = Capsule(content=idea, perspective="test")
        novelty = cayde.memory.check_temporal_novelty(test_capsule)
        status = "🚫 ANACHRONISM" if novelty['anachronism_detected'] else "✅ TEMPORALLY OK"
        print(f"   {idea}: {status} - {novelty['novelty_assessment']}")

    print("\n📊 Final Memory State with Time Awareness:")
    print(f"Total capsules: {len(cayde.memory.capsules)}")
    high_confidence = sum(1 for c in cayde.memory.capsules if c.certainty > 0.8)
    temporal_penalty = sum(1 for c in cayde.memory.capsules if 0.1 < c.certainty <= 0.4)
    severely_penalized = sum(1 for c in cayde.memory.capsules if c.certainty <= 0.1)
    print(f"High confidence (certainty > 0.8): {high_confidence}")
    print(f"Temporally penalized (0.1 < certainty ≤ 0.4): {temporal_penalty}")
    print(f"Severely penalized (certainty ≤ 0.1): {severely_penalized}")

    print("\n🎯 Time Awareness Test Complete!")
    print("Cayde now experiences the future as unknown and prevents anachronistic reasoning!")
    return cayde


def demo_roca_orbital():
    """Run a demonstration of the ROCA orbital memory system with Cayde experiencing Einstein's trajectory"""
    print("🚀 ROCA Orbital Memory System Demo with Cayde Personality")
    print("🎯 Experiencing Einstein's Intellectual Trajectory")
    print("=" * 70)

    # Create memory system
    memory = RocaOrbitalMemory()

    # Einstein's intellectual trajectory - chronological learning sequence
    einstein_trajectory = [
        # Early education and first insights (1895-1900)
        {
            "title": "Early Education - Classical Physics",
            "content": "Classical mechanics and electromagnetism fundamentals",
            "kind": CapsuleKind.CONCEPT,
            "initial_status": "success",
            "initial_confidence": 0.8,
            "perspective": "learning"
        },
        {
            "title": "Thought Experiment: Riding Light Beam",
            "content": "Imagining riding alongside a light beam - leads to relativity insight",
            "kind": CapsuleKind.OBSERVATION,
            "initial_status": "failure",  # Initially confusing
            "initial_confidence": 0.3,
            "perspective": "personal_reflection"
        },
        # Annus Mirabilis (1905)
        {
            "title": "Special Relativity Paper",
            "content": "Special theory of relativity - constancy of light speed",
            "kind": CapsuleKind.THEORY,
            "initial_status": "failure",  # Initially rejected by establishment
            "initial_confidence": 0.6,
            "perspective": "revolutionary"
        },
        {
            "title": "Photoelectric Effect",
            "content": "Light behaves as particles (photons) with E = hf",
            "kind": CapsuleKind.THEORY,
            "initial_status": "failure",  # Contradicted wave theory
            "initial_confidence": 0.4,
            "perspective": "experimental"
        },
        {
            "title": "E = mc² Derivation",
            "content": "Equivalence of mass and energy",
            "kind": CapsuleKind.THEORY,
            "initial_status": "failure",  # Seemed abstract and useless
            "initial_confidence": 0.5,
            "perspective": "mathematical"
        },
        # Post-1905 struggles
        {
            "title": "General Relativity Struggle",
            "content": "Attempting to generalize relativity to accelerating frames",
            "kind": CapsuleKind.THEORY,
            "initial_status": "failure",  # Mathematical difficulties
            "initial_confidence": 0.2,
            "perspective": "frustrated"
        },
        {
            "title": "Equivalence Principle Insight",
            "content": "Gravitational and inertial mass equivalence",
            "kind": CapsuleKind.CONCEPT,
            "initial_status": "success",
            "initial_confidence": 0.7,
            "perspective": "breakthrough"
        },
        # 1915-1917: Major breakthroughs
        {
            "title": "General Relativity Completion",
            "content": "Field equations of gravitation - geometry of spacetime",
            "kind": CapsuleKind.THEORY,
            "initial_status": "failure",  # Too complex for contemporaries
            "initial_confidence": 0.8,
            "perspective": "triumphant"
        },
        {
            "title": "Cosmological Constant",
            "content": "Lambda term to make universe static",
            "kind": CapsuleKind.THEORY,
            "initial_status": "failure",  # Later called biggest blunder
            "initial_confidence": 0.6,
            "perspective": "speculative"
        },
        # Later years: Unified theory quest
        {
            "title": "Unified Field Theory Quest",
            "content": "Attempting to unify gravity and electromagnetism",
            "kind": CapsuleKind.THEORY,
            "initial_status": "failure",  # Never succeeded
            "initial_confidence": 0.3,
            "perspective": "unsatisfied"
        },
        {
            "title": "Quantum Mechanics Skepticism",
            "content": "Rejection of quantum uncertainty - God doesn't play dice",
            "kind": CapsuleKind.OBSERVATION,
            "initial_status": "failure",  # Went against emerging consensus
            "initial_confidence": 0.4,
            "perspective": "philosophical"
        }
    ]

    # Experience Einstein's trajectory progressively
    memory.memory.progressive_learning_session("Einstein", einstein_trajectory)

    # Add modern validations to show theories proven later
    print("\n🧪 Adding historical validations...")
    memory.add_capsule("1919 Eclipse confirms General Relativity", character="Eddington",
                      kind=CapsuleKind.OBSERVATION, success_status="success")
    memory.add_capsule("Cosmological Constant explains dark energy (1998)", character="Riess",
                      kind=CapsuleKind.OBSERVATION, success_status="success")
    memory.add_capsule("E = mc² enables nuclear energy", character="Modern_Physics",
                      kind=CapsuleKind.THEORY, success_status="success")

    # Allow final integration
    print("\n🔄 Final integration phase...")
    for _ in range(5):
        memory.update_memory()

    # Show final Cayde reflection
    final_reflection = memory.memory.cayde_reflect_on_scientist("Einstein")
    print(f"\n{final_reflection}")

    # Generate trajectory-inspired hypotheses
    print("\n🧠 Cayde generating hypotheses inspired by Einstein's trajectory...")
    hypotheses = memory.memory.generate_hypotheses(3)
    for i, hyp in enumerate(hypotheses, 1):
        print(f"Hypothesis {i}: {hyp.content}")

    # Run visualization
    print("\nStarting orbital visualization...")
    print("Colors: Cyan=Cayde, Yellow=Insights, Purple=Theories Proven Later")
    print("Watch how confidence evolves and conflicts resolve over time!")
    memory.visualize()

if __name__ == "__main__":
    # Run the full ROCA orbital demo with Cayde's complete system
    demo_roca_orbital()
