# IJsselbridge Python model

IJssel bridge twin-girder finite element model in Python. Based on the 4-DOF
Euler-Bernoulli beam element formulation. The Python version of the model loads
the pre-computed stiffness matrices and model properties, stored in `model/model_data/`
and returns an instance of the model class. This avoids the dependency on Julia but
limits the flexibility of the model, since the sensor locations and discretization are
pre-determined.

## Usage


Model properties are loaded from `model/model_data/` upon initialization
based on the specified parameters.

The `il_stress_truckload(c, lane, Kr = None, Kv = None)` method returns the stress
influence line (in MPa) for a truck load on a specified lane, where:

* `c`: Scalar slope of the lateral load function `f(c,z) = c * (z - 2.85) + 0.5`.
* `lane`: String specifying the loaded lane. Can be `"left"` or `"right"`.
* `Kr`: Optional numpy array, rotational spring stiffness at supports in the order
`[Kr_1_right, Kr_1_left, Kr_2_right, Kr_2_left, ... ]` (in kNm/rad).

* `Kv`: Optional scalar, stiffness of the vertical springs representing the K-braces
  connecting the two girders (in kN/m).

If both `Kr` and `Kv` are `None`, the last calculated influence line is scaled according
to the specified `c`.

```Python
import numpy as np
from model.model_twin_girder_betti import IJssel_bridge_model

# Set model parameters
meas_type = "fugro"             # Determines the load. Can be "TNO" or "fugro"
E = 210e6                       # Elasticity modulus in kPa
max_elem_length = 2.0 * 1e3     # Maximum element length in mm
sname = "H1_S"                  # Sensor label

model = IJssel_bridge_model(
    sname, E, max_elem_length=max_elem_length, truck_load=meas_type
)

y = model.il_stress_truckload(-0.1, "left", Kr=np.zeros(12), Kv=0.0)
```
