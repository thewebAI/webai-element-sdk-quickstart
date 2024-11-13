# webAI Element SDK Quickstart

Follow this guide to get up and running with your own custom elements.

## Installation/Environment

Currently, elements support two methods of environment setup.

Folder structure
```
- element-name
    - element_name
        - __init__.py
    - publish.json (will be generated by the `builder` utility)

    - requirements.txt
    - setup.py
```

### uv pip

requirements.txt
```
webai_element_sdk>=x.y.z
```

setup.py
```python
from setuptools import find_packages, setup

setup(name="element_name", packages=find_packages())
```


## Boilerplate

The following is basic element with a single input and output. This element would operate in the middle of a flow. If you need an element at the start or end of a flow, remove the Inputs or Outputs class, respectively.

```python
import time
from uuid import UUID

from webai_element_sdk import Context, Element
from webai_element_sdk.comms.messages import ColorFormat, Frame
from webai_element_sdk.element.variables import ElementOutputs, ElementInputs, Input, Output


class Inputs(ElementInputs):
    default_in1 = Input[Frame]()
    default_in2 = Input[AsyncIterator[Frame]]()


class Outputs(ElementOutputs):
    default_out = Output[Frame]()


element = Element(
    id=UUID("<uuid>"),
    name="element_name",
    description="Outputs a simple start message to kick-off looping or cycle flows",
    display_name="Element Name",
    version="0.0.1",
    inputs=Inputs(),
    outputs=Outputs(),
)


@element.startup
async def startup(ctx: Context[Inputs, Outputs, None]):
    print(f"Starting...")


@element.shutdown
async def shutdown(ctx: Context[Inputs, Outputs, None]):
    print(f"Shutting down...")


@element.executor
async def run(ctx: Context[Inputs, Outputs, None]):
    print(f"Running...")

    # retrieve a single Frame per execution
    input_value: Frame = ctx.inputs.default_in1.value

    # retrieve a generator to loop over
    input_generator: AsyncIterator[Frame] = ctx.inputs.default_in2.value

    for frame in input_generator:
        # ... do some work

    # ... do some work

    output_value: Frame = Frame(
        ndframe=None, # numpy image
        rois=[],
        other_data={"abc": 123},
    )

    yield ctx.outputs.default_out(output_value)
```

See the `examples` folder for more information.

## Settings

Settings allow your element to be configured prior to running in a flow. The following are examples of the available settings:

```python
# ... other imports

from webai_element_sdk import Context, Element
from webai_element_sdk.element.settings import (
    ElementSettings,
    BoolSetting,
    NumberSetting,
    TextSetting,
)

class Settings(ElementSettings):
    setting1 = TextSetting(
        name="setting1",
        display_name="Setting 1",
        valid_values=["a", "b", "c"], # optional
        default="b",
        hints=["dropdown"] # optional
    )
    setting2 = NumberSetting[float]( # or 'int'
        name="setting2",
        display_name="Setting 2",
        default=0.4,
        min_value=0,
        max_value=1,
    )
    setting3 = BoolSetting(
        name="setting3",
        display_name="Setting 3",
        default=True
    )


# ... inputs and outputs

element = Element(
    id=UUID("<uuid>"),
    name="element_name",
    description="Outputs a simple start message to kick-off looping or cycle flows",
    display_name="Element Name",
    version="0.0.1",
    settings=Settings(),
    inputs=Inputs(),
    outputs=Outputs(),
)

@element.executor
async def run(ctx: Context[Inputs, Outputs, Settings]):
    print(f"Running...")

    setting1: str = ctx.settings.setting1.value
    setting2: float = ctx.settings.setting2.value
    setting3: bool = ctx.settings.setting3.value

    input_value: Frame = ctx.inputs.default_in.value

    # ... do some work

    output_value: Frame = Frame(
        ndframe=None, # numpy image
        rois=[],
        other_data={"abc": 123},
    )

    yield ctx.outputs.default_out(output_value)

# ... other lifecycle hooks
```


## Packaging and Importing Utility

There is a standalone binary tool that can be used to package your element prior to installation. This can be found in the Application Support folder of your Navigator installation: `/Applications/Navigator.app/Contents/Resources/support/builder`

### Metadata Generation

Use `builder` to generate the `publish.json` file. This is required prior to packaging the element for import. Re-run this if your element metadata changes (i.e. version, settings, inputs/outputs, etc).

`./path/to/builder generate path/to/element-name`

### Element Packaging

Use `builder` to package your element for import.

`./path/to/builder package --path path/to/dest path/to/element-name`

### Element Import

This imports an official version into your local element registry.

Ensure that Navigator/Runtime controller is running. Then run:

`./path/to/builder import abs/path/to/packaged-zip.zip`

### Element Dev Import

This imports an dev version of your element that links to the source code of your element so you can continue editing without updating the version and re-packaging.

Ensure that Navigator/Runtime controller is running. Then run:

`./path/to/builder dev-import path/to/element-name`

### Navigator Element Drawer Import

Coming Soon

## Examples

### send -> relay -> receive

This is a very simple example of elements that pass image frames through a flow when connected on a canvas. Use this as examples to see how frame sending works.

### ai trainer, predictor

These elements are given as examples of 3 things:
- how training and inference work
  - logging training metrics
  - using common model code
- artifact saving and loading
  - register an artifact automatically by saving it in the training element's working directory. Runtime does the rest.
  - by using a special artifact setting in the inference element, an artifact will be injected into the current working directory
- parent/child element relationship
  - look at the top level init file of `simple_ai`
  - it makes sense to group training and inference children elements under a parent to share common code and dependencies
  - the parent element id is also how an inference element knows which artifacts it has access to