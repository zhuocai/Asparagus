import sympy as sp

sp_unit = 1


class Polynomial:
    def __init__(self, monomial_dict):
        """ 
        Input: monomial_dict[monomial] = coefficient
        """
        self.monomial_dict = monomial_dict
        self.total_degree = self.get_total_degree()

    def subs(self, substitute_dict):
        """
        The function should return a new polynomial obtained after substitution ONLY in the monomials
        No substitution allowed in the coefficients
        """
        new_monomial_dict = {}
        for mono in self.monomial_dict:
            if mono == 1:
                new_mono = mono
            else:
                new_mono = mono.subs(substitute_dict)
            new_monomial_dict[new_mono] = self.monomial_dict[mono]

        return Polynomial(new_monomial_dict)

    def subs_complex(self, final_values):
        """
        final_values is a dict that map each variable to a Polynomial
        The function should return a new polynomial obtained after substitution ONLY in the monomials
        No substitution allowed in the coefficients
        """
        net_poly = Polynomial({})
        new_monomial_dict = {}
        for mono in self.monomial_dict:
            if mono == 1:
                new_poly = Polynomial({sp_unit: self.monomial_dict[mono]})
                net_poly += new_poly
            else:
                local_poly = Polynomial({sp_unit: self.monomial_dict[mono]})
                for fac in mono.as_ordered_factors():
                    if fac.is_number:
                        expr = fac
                    else:
                        var, power = fac.as_base_exp()
                        expr = final_values[str(var)]**power
                    local_poly = local_poly*(expr)
                net_poly += local_poly
        return net_poly

    def as_expr(self):
        base = 0
        for mono in self.monomial_dict:
            coeff = self.monomial_dict[mono]
            base += coeff*mono
        return base

    def total_degree(self):
        return self.total_degree

    def get_total_degree(self):
        largest_deg = 0
        for mono in self.monomial_dict:
            if mono == 1:
                continue
            largest_deg = max(largest_deg, mono.as_poly().total_degree())
        return largest_deg

    def __add__(self, other):
        new_monomial_dict = {}
        for mono in self.monomial_dict:
            new_monomial_dict[mono] = self.monomial_dict[mono]
        for mono in other.monomial_dict:
            if mono in new_monomial_dict:
                new_monomial_dict[mono] += other.monomial_dict[mono]
            else:
                new_monomial_dict[mono] = other.monomial_dict[mono]
        return Polynomial(new_monomial_dict)

    def __sub__(self, other):
        neg_monomial_dict = {}
        for mono in other.monomial_dict:
            neg_monomial_dict[mono] = -other.monomial_dict[mono]
        neg_poly = Polynomial(neg_monomial_dict)
        return Polynomial.__add__(self, neg_poly)

    def __pow__(self, power):
        base = Polynomial({sp_unit: 1})
        for i in range(power):
            base *= self
        return base

    def __mul__(self, other):
        if type(other) != type(self):
            # other should be multiplied like a number and added to the coefficient
            if type(other) == str:
                other = sp.Symbol(other)
            new_monomial_dict = {}
            for mono in self.monomial_dict:
                coeff1 = self.monomial_dict[mono]
                new_monomial_dict[mono] = coeff1*other
            return Polynomial(new_monomial_dict)

        new_monomial_dict = {}
        for mono1 in self.monomial_dict:
            coeff1 = self.monomial_dict[mono1]
            if coeff1 == 0:
                continue
            for mono2 in other.monomial_dict:
                coeff2 = other.monomial_dict[mono2]
                if coeff2 == 0:
                    continue
                new_mono = mono1*mono2
                if new_mono in new_monomial_dict:
                    new_monomial_dict[new_mono] += coeff1*coeff2
                else:
                    new_monomial_dict[new_mono] = coeff1*coeff2
        new_poly = Polynomial(new_monomial_dict)
        return new_poly

    def __truediv__(self, other):
        if sp_unit not in other.monomial_dict or len(other.monomial_dict) != 1:
            raise Exception("Polynomial can divided only by a number")
        num = other.monomial_dict[sp_unit]
        return self*Polynomial({sp_unit: 1/num})

    def get_eq_zero_conditions(self):
        monomial_list = []
        sub_QP = []
        for mono in self.monomial_dict:
            monomial_list.append(mono)
            coefficient = self.monomial_dict[mono]
            sub_QP.append(sp.Equality(coefficient, 0))
        return sub_QP, monomial_list

    def to_string(self):
        lst = []
        for mono in self.monomial_dict:
            lst.append("({})*({})".format(self.monomial_dict[mono], mono))
        return "+".join(lst)

    def to_string_nonzero(self):
        lst = []
        for mono in self.monomial_dict:
            coef = self.monomial_dict[mono]
            # print('mono',mono, 'coef', coef)
            if coef != 0:  # this is ineffective
                if mono != sp_unit:
                    lst.append("({})*({})".format(coef, mono))
                else:
                    lst.append(str(coef))
        return lst

    def __repr__(self):
        """
        method to return the canonical string representation 
        of a polynomial.

        """
        return "Polynomial(" + str(self.monomial_dict) + ")"

    def __str__(self):
        """
        method to return the canonical string representation 
        of a polynomial.

        """
        return "Polynomial(" + str(self.monomial_dict) + ")"
