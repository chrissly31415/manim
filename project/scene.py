from manim import *
import numpy as np

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        text = Text("hallo")
        self.add(text)

        square = Square()  # create a square
        square.animate.rotate(PI / 4)  # rotate a certain amount
        square.next_to(circle, RIGHT, buff=0.5) 

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation

        left_square = Square(color=BLUE, fill_opacity=0.7).shift(2 * LEFT)
        right_square = Square(color=GREEN, fill_opacity=0.7).shift(2 * RIGHT)
        self.play(
            left_square.animate.rotate(PI), Rotate(right_square, angle=PI), run_time=2
        )
        self.wait()

        #self.play(Create(circle))  # s


class HarmonicOscillator(Scene):
    def construct(self):
        # Parameters
        amplitude = 2          # Amplitude of oscillation
        frequency = 0.5        # Frequency of oscillation
        omega = 2 * np.pi * frequency
        mass_value = 1         # Assume mass (m) = 1 for simplicity
        spring_constant = 1    # Spring constant (k)
        time_tracker = ValueTracker(0)  # Track time manually

        # Create objects
        mass = Square(side_length=0.4, fill_color=BLUE, fill_opacity=1)
        mass.shift(amplitude * UP)  # Start vertically at max amplitude
        spring = Line(ORIGIN, mass.get_bottom()).set_stroke(width=2)
        ceiling = Line(3 * LEFT, 3 * RIGHT).shift(3 * UP)  # Fixed ceiling

        # Position labels
        self.add(ceiling, spring, mass)

        def update_spring(spring):
            # Update the spring to connect the ceiling and the mass as it oscillates
            spring.put_start_and_end_on(ceiling.get_bottom(), mass.get_top())

        spring.add_updater(update_spring)

        # Define oscillation update function for the mass
        def oscillate(mob, dt):
            time_tracker.increment_value(dt)
            y_position = amplitude * np.cos(omega * time_tracker.get_value())
            mob.move_to(y_position * UP)

        mass.add_updater(oscillate)

        # Define energy label updates
        potential_energy_label = MathTex("U = \\frac{1}{2}kx^2")
        kinetic_energy_label = MathTex("K = \\frac{1}{2}mv^2")
        potential_energy_label.animate.to_corner(UL).shift(DOWN)
        kinetic_energy_label.to_corner(UR).shift(DOWN)

        

        def update_potential_energy(label):
            x = amplitude * np.cos(omega * time_tracker.get_value())
            U = 0.5 * spring_constant * x**2
            label.become(MathTex(f"U = {U:.2f}").to_corner(UL).shift(DOWN))

        def update_kinetic_energy(label):
            x = amplitude * np.cos(omega * time_tracker.get_value())
            K = 0.5 * mass_value * omega**2 * (amplitude**2 - x**2)
            label.become(MathTex(f"K = {K:.2f}").to_corner(UR).shift(DOWN))

    
        # Add labels to the scene
        self.add(potential_energy_label, kinetic_energy_label)

        self.play(
            FadeIn(potential_energy_label) , FadeIn(kinetic_energy_label) ,run_time=5
        )
        potential_energy_label.add_updater(update_potential_energy)
        kinetic_energy_label.add_updater(update_kinetic_energy)
        self.wait(5) 
        # Play animation
        self.play(FadeIn(mass), FadeIn(spring))
        self.wait(5)  # Duration of the animation


class VectorArrow(Scene):
    def construct(self):
        dot = Dot(ORIGIN)
        arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
        numberplane = NumberPlane()
        origin_text = Text('(0, 0)').next_to(dot, DOWN)
        tip_text = Text('(2, 2)').next_to(arrow.get_end(), RIGHT)
        self.add(numberplane, dot, arrow, origin_text, tip_text)


class Sets(Scene):
    def construct(self):
        # Create the ellipses
        ellipse1 = Ellipse(
            width=4.0, height=5.0, fill_opacity=0.5, color=BLUE, stroke_width=10
        ).move_to(LEFT)
        ellipse2 = ellipse1.copy().set_color(color=RED).move_to(RIGHT)
        
        # Add labels to each ellipse
        left_label = Text("Metal", font_size=24).next_to(ellipse1, UP)
        right_label = Text("HipHop", font_size=24).next_to(ellipse2, UP)

        # Boolean operations header
        bool_ops_text = MarkupText("<u>Set Operation</u>").next_to(ellipse1, UP * 3)
        
        # Group the ellipses and labels together for easy positioning
        ellipse_group = Group(bool_ops_text, ellipse1, ellipse2, left_label, right_label).move_to(LEFT * 3)
        self.play(FadeIn(ellipse_group))

        # Intersection with label
        i = Intersection(ellipse1, ellipse2, color=GREEN, fill_opacity=0.5)
        #intersection_label = Text("Couples", font_size=23).next_to(i, UP)
        self.play(i.animate.scale(0.25).move_to(RIGHT * 5 + UP * 2.5))
        #self.play(FadeIn(intersection_label))

        # Union with label
        u = Union(ellipse1, ellipse2, color=ORANGE, fill_opacity=0.5)
        union_text = Text("Union", font_size=23)
        self.play(u.animate.scale(0.3).next_to(i, DOWN, buff=union_text.height * 3))
        union_text.next_to(u, UP)
        self.play(FadeIn(union_text))

        # Exclusion with label
        e = Exclusion(ellipse1, ellipse2, color=YELLOW, fill_opacity=0.5)
        exclusion_text = Text("Exclusion", font_size=23)
        self.play(e.animate.scale(0.3).next_to(u, DOWN, buff=exclusion_text.height * 3.5))
        exclusion_text.next_to(e, UP)
        self.play(FadeIn(exclusion_text))

        # Difference with label
        d = Difference(ellipse1, ellipse2, color=PINK, fill_opacity=0.5)
        difference_text = Text("Difference", font_size=23)
        self.play(d.animate.scale(0.3).next_to(u, LEFT, buff=difference_text.height * 3.5))
        difference_text.next_to(d, UP)
        self.play(FadeIn(difference_text))


class VectorField(Scene):
    def construct(self):
        func = lambda pos: np.sin(pos[1] / 2) * RIGHT + np.cos(pos[0] / 2) * UP
        vector_field = ArrowVectorField(
            func, x_range=[-7, 7, 1], y_range=[-4, 4, 1], length_func=lambda x: x / 2
        )
        self.add(vector_field)


class SinusCurve(Scene):
    # contributed by heejin_park, https://infograph.tistory.com/230
    def construct(self):
        self.show_axis()
        self.show_circle()
        self.move_dot_and_draw_curve()
        self.wait()

    def show_axis(self):
        x_start = np.array([-6,0,0])
        x_end = np.array([6,0,0])

        y_start = np.array([-4,-2,0])
        y_end = np.array([-4,2,0])

        x_axis = Line(x_start, x_end)
        y_axis = Line(y_start, y_end)

        self.add(x_axis, y_axis)
        self.add_x_labels()

        self.origin_point = np.array([-4,0,0])
        self.curve_start = np.array([-3,0,0])

    def add_x_labels(self):
        x_labels = [
            MathTex("\pi"), MathTex("2 \pi"),
            MathTex("3 \pi"), MathTex("4 \pi"),
        ]

        for i in range(len(x_labels)):
            x_labels[i].next_to(np.array([-1 + 2*i, 0, 0]), DOWN)
            self.add(x_labels[i])

    def show_circle(self):
        circle = Circle(radius=1)
        circle.move_to(self.origin_point)
        self.add(circle)
        self.circle = circle

    def move_dot_and_draw_curve(self):
        orbit = self.circle
        origin_point = self.origin_point

        dot = Dot(radius=0.08, color=YELLOW)
        dot.move_to(orbit.point_from_proportion(0))
        self.t_offset = 0
        rate = 0.25

        def go_around_circle(mob, dt):
            self.t_offset += (dt * rate)
            # print(self.t_offset)
            mob.move_to(orbit.point_from_proportion(self.t_offset % 1))

        def get_line_to_circle():
            return Line(origin_point, dot.get_center(), color=BLUE)

        def get_line_to_curve():
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            return Line(dot.get_center(), np.array([x,y,0]), color=YELLOW_A, stroke_width=2 )


        self.curve = VGroup()
        self.curve.add(Line(self.curve_start,self.curve_start))
        def get_curve():
            last_line = self.curve[-1]
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            new_line = Line(last_line.get_end(),np.array([x,y,0]), color=YELLOW_D)
            self.curve.add(new_line)

            return self.curve

        dot.add_updater(go_around_circle)

        origin_to_circle_line = always_redraw(get_line_to_circle)
        dot_to_curve_line = always_redraw(get_line_to_curve)
        sine_curve_line = always_redraw(get_curve)

        self.add(dot)
        self.add(orbit, origin_to_circle_line, dot_to_curve_line, sine_curve_line)
        self.wait(8.5)

        dot.remove_updater(go_around_circle)



class MatrixTest(Scene):
    def construct(self):
        
        # Title text "Matrix Multiplication" with styling
        title = Text("Matrix Multiplication", font_size=64, color=BLUE).to_edge(UP)

        # Display the title with a fancy animation
        self.play(FadeIn(title, scale=1.2), run_time=2)
        self.wait(1)
        self.play(FadeOut(title))

        # Define the matrices A and B, and calculate their product C
        A = np.array([[-1, 5], [7, 11]])
        B = np.array([[2, 3], [-8, 0]])
        C = np.dot(A, B)

        # Convert matrices to Manim's Matrix objects
        matrixA = Matrix(A, h_buff=1.5).set_color(PURPLE).scale(0.8)
        matrixB = Matrix(B, h_buff=1.5).set_color(RED).scale(0.8)
        matrixC = Matrix(C, h_buff=1.5).set_color(PURE_GREEN).scale(0.8)

        # Set up the layout for displaying the matrices
        matrixA.to_corner(UP + LEFT)
        dot_symbol = MathTex(r"\cdot", font_size=80).next_to(matrixA, RIGHT)
        matrixB.next_to(dot_symbol, RIGHT)
        equals_symbol = MathTex("=", font_size=80).next_to(matrixB, RIGHT)
        matrixC.next_to(equals_symbol, RIGHT)

        # Display matrices and symbols in initial layout
        self.play(Write(matrixA), Write(dot_symbol), Write(matrixB), Write(equals_symbol))
        self.wait(1)

        # Loop through each element in the result matrix
        for i in range(2):  # Row of A
            for j in range(2):  # Column of B
                # Highlight the current row in A and column in B
                row_highlight = SurroundingRectangle(matrixA.get_rows()[i], color=BLUE, buff=0.1)
                col_highlight = SurroundingRectangle(matrixB.get_columns()[j], color=YELLOW, buff=0.1)
                self.play(Create(row_highlight), Create(col_highlight))

                # Calculate and display the summation process for each element
                result_value = C[i][j]
                step_texts = [
                    MathTex(f"{A[i][0]} \\times {B[0][j]}"),
                    MathTex("+"),
                    MathTex(f"{A[i][1]} \\times {B[1][j]}"),
                    MathTex("="),
                    MathTex(f"{result_value}")
                ]

                # Arrange the steps in a horizontal layout under the matrices
                steps = VGroup(*step_texts).arrange(RIGHT, buff=0.5).next_to(matrixB, DOWN, buff=1.5)
                self.play(Write(steps[0]), Write(steps[1]), Write(steps[2]))
                self.wait(1)
                self.play(Write(steps[3]), Write(steps[4]))
                self.wait(1)

                # Place result in matrix C and fade out the step calculations
                self.play(Transform(steps[4].copy(), matrixC.get_entries()[i * 2 + j]))
                self.play(FadeOut(steps), FadeOut(row_highlight), FadeOut(col_highlight))

        # Display the final result matrix
        self.play(Write(matrixC))
        self.wait(2)

        # End screen text
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        end_text = Text("Matrix x Matrix Multiplication:", font_size=40, color=BLUE).move_to(ORIGIN)
        self.play(Write(end_text))
        self.wait(2)


        # Formula for matrix multiplication at the end
        formula = MathTex(
            r"(AB)_{ij} = \sum_{k=1}^n A_{ik} \cdot B_{kj}", 
            font_size=48, color=WHITE
        )
        self.play(FadeOut(end_text))

        self.play(Write(formula))
        # Show the formula with a fade-in animation
        self.wait(2)
        # Zoom in on the formula
        self.play(
            ScaleInPlace(formula, 2))
        
        self.wait(2)



class BinarySearch(Scene):
    def construct(self):
        # Define the array elements and initial positions
        elements = [4, 10, 16, 24, 32, 46, 76, 112, 144, 182]
        target = 46
        
        # Initial positions for each row
        rows = []
        
        # Display initial array at the top
        self.display_array(elements, 0, "Binary Search for 46", show_indices=True)
        
        # Step 1: Initial midpoint check, take upper half
        rows.append((0, 9, 4, "Interval mid point: 32<46"))
        self.display_array(elements, 1, rows[-1][3], low=rows[-1][0], high=rows[-1][1], mid=rows[-1][2])
        
        # Step 2: Update low, new midpoint check, take lower half
        rows.append((5, 9, 7, "Interval mid point: 112>46"))
        self.display_array(elements, 2, rows[-1][3], low=rows[-1][0], high=rows[-1][1], mid=rows[-1][2])
        
        # Step 3: Update high, final midpoint check, found target
        rows.append((5, 6, 5, "Found 46 at Index 5"))
        self.display_array(elements, 3, rows[-1][3], low=rows[-1][0], high=rows[-1][1], mid=rows[-1][2])
        
        # Hold final result for a moment
        self.wait(5)
    
    def display_array(self, elements, row_num, action_text, low=None, high=None, mid=None, show_indices = False):
        # Create row of boxes and numbers
        boxes = VGroup(*[Square().scale(0.3).set_fill(BLUE, opacity=0.2) for _ in elements])
        numbers = VGroup(*[Text(str(num)).scale(0.5) for num in elements])
        
        # Arrange boxes and numbers for this row
        boxes.arrange(RIGHT, buff=0.1).shift(UP * (2.5 - row_num * 1.5))  # Adjust row positioning
        for box, number in zip(boxes, numbers):
            number.move_to(box.get_center())
        
        # Add index labels below the array
        
        indices = VGroup(*[Text(str(i)).scale(0.3) for i in range(len(elements))])
        if show_indices:
            for i, index in enumerate(indices):
                index.next_to(boxes[i], DOWN)
        
        # Add action text above the array
        action = Text(action_text, color=WHITE).scale(0.4).next_to(boxes, LEFT)
        
        # Add highlighting if low, high, and mid are provided
        if low is not None:
            boxes[low].set_fill(GREEN, opacity=0.5)
        if high is not None:
            boxes[high].set_fill(RED, opacity=0.5)
        if mid is not None:
            boxes[mid].set_fill(YELLOW, opacity=0.5)
        
        # Show this row with highlights
        if show_indices:
            self.play(FadeIn(boxes), Write(numbers), Write(indices), Write(action))
        else:
            self.play(FadeIn(boxes), Write(numbers), Write(action))
        # Add bracket to show current interval (low to high)
        if low is not None and high is not None:
            interval_brace = Brace(VGroup(boxes[low], boxes[high]), DOWN, color=WHITE)
            interval_text = Text(f"Interval: [{low}, {high}]").scale(0.2).next_to(interval_brace, DOWN).shift(UP*0.1)
            self.play(FadeIn(interval_brace), Write(interval_text))
        
        self.wait(1)





