(potential-lasp)=

# lasp

The `lasp` provider loads LASP neural-network potential files.

## Requirements

Provide the LASP executable and trained potential files for every element in the calculation.

## Configuration

```yaml
potential:
  provider: lasp
  parameters:
    command: lasp
    type_list: [Cu, O]
    model: [./Cu.pot, ./O.pot]
```

`model` accepts a path or list of existing files. When the number of models
matches `type_list`, GDPy maps files to elements in that order. Otherwise the
first model file is assigned to every element, so use that form only for a
file that covers all requested species.
