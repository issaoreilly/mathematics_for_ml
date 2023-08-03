import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import itertools
from scipy.linalg import null_space

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Interactive 3D Plot"),
    dcc.Input(id="coefficients1", type="text", value="10,1,1,0"),
    dcc.Input(id="coefficients2", type="text", value="1,10,1,5"),
    dcc.Input(id="coefficients3", type="text", value="1,1,10,-3"),
    dcc.Graph(id="graph")
])

# Define the callback to update the graph.


@app.callback(
    Output('graph', 'figure'),
    [Input('coefficients1', 'value'), Input(
        'coefficients2', 'value'), Input('coefficients3', 'value')]
)
def update_graph(coefficients1_str, coefficients2_str, coefficients3_str):
    coefficients_list = [coefficients1_str,
                         coefficients2_str, coefficients3_str]
    coefficients = np.array([[float(n.strip()) for n in coeff_str.split(",")]
                            for coeff_str in coefficients_list])

    # Define ranges for x and y
    x = np.linspace(-50, 50, 50)
    y = np.linspace(-50, 50, 50)
    z = np.linspace(-5, 5, 10)

    data = []

    # Calculate Z values for each plane and plot
    for index, coeff in enumerate(coefficients):
        # A * X + B * Y + C * Z = 0
        A, B, C, D = coeff

        if A != 0:
            # Create meshgrid for y, z
            Y, Z = np.meshgrid(y, z)
            X = (-B * Y - C * Z + D) / A
        if B != 0:
            # Create meshgrid for x, z
            X, Z = np.meshgrid(x, z)
            Y = (-A * X - C * Z + D) / B
        if C != 0:
            # Create meshgrid for x, y
            X, Y = np.meshgrid(x, y)
            Z = (-A * X - B * Y + D) / C

        # Add 3d surface to data
        data.append(go.Surface(x=X, y=Y, z=Z, opacity=0.5))

        # Solve system of equations to find intersection lines
    # For each pair of coefficient arrays
    for pair in itertools.combinations(coefficients, 2):
        pair = np.array(pair)  # Convert pair to a 2D array

        # Solve the system of equations defined by the pair of planes
        solution, residuals, rank, s = np.linalg.lstsq(
            pair[:, :3], pair[:, 3], rcond=None)

        if rank == 2:  # If the rank of the coefficient matrix is 2
            # The planes intersect along a line
            # The direction vector of the line is the null space of the coefficient matrix
            # Calculate the null space of the coefficient matrix
            null_space_vector = null_space(pair[:, :3])

            # We plot the line as a point plus a scalar times the direction vector
            # For 100 equally spaced points from -10 to 10
            points_on_line = []
            for t in np.linspace(-500, 500, 1000):
                # Calculate a point on the line of intersection
                point_on_line = solution + t * null_space_vector.ravel()
                points_on_line.append(point_on_line)
                # Add line of intersection to data

            # Transpose the array to get X, Y, Z arrays
            points_on_line = np.array(points_on_line).T
            # Add line of intersection to data
            # Add the line to the data list as a 3D line plot
            data.append(go.Scatter3d(x=points_on_line[0],
                                     y=points_on_line[1],
                                     z=points_on_line[2],
                                     mode='lines',
                                     # Set color to black and width to 3
                                     line=dict(color='black', width=3)
                                     ))

    # Solve system of equations to find intersection point
    try:
        # np.linalg.solve returns the solution to the system of linear equations given by the coefficient matrix and the constant vector
        intersection_point = np.linalg.solve(
            coefficients[:, :3], coefficients[:, 3])
        # Add the intersection point to the data list as a 3D scatter plot
        data.append(go.Scatter3d(x=[intersection_point[0]], y=[intersection_point[1]], z=[
                    intersection_point[2]], mode='markers', marker=dict(size=8, color='red')))
    except np.linalg.LinAlgError:
        # The planes do not intersect at a single point
        pass

    fig = go.Figure(data=data)  # Create a figure from the data list

    # Set labels and adjust dimensions and aspect ratio
    fig.update_layout(scene=dict(xaxis_title='X',  # Set the title of the x-axis
                                 yaxis_title='Y',  # Set the title of the y-axis
                                 zaxis_title='Z',  # Set the title of the z-axis
                                 # range for x-axis
                                 xaxis=dict(range=[-70, 70]),
                                 # range for y-axis
                                 yaxis=dict(range=[-70, 70]),
                                 # range for z-axis
                                 zaxis=dict(range=[-70, 70]),

                                 aspectmode='auto'),  # Adjust the aspect ratio to be automatic
                      width=1000,    # Set the width of the figure
                      height=600,    # Set the height of the figure
                      margin=dict(r=20, b=10, l=10, t=10))  # Adjust the margins of the figure

    return fig  # Return the figure


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)  # Run the Dash app in debug mode
