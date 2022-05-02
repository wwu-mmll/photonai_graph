# Graph Controllability

## Graph Controllability Transformation

::: photonai_graph.Controllability.controllability_measures.ControllabilityMeasureTransform.__init__
    rendering:
        show_root_toc_entry: False

### Extraction of Graph Controllability measures
Instead of using the controllability measures in a PHOTONAI pipeline, you are also able to 
extract the measures with PHOTONAI Graph and generate a CSV file for use with third party software.

```python
from photonai_graph.Controllability.controllability_measures import ControllabilityMeasureTransform

transform = ControllabilityMeasureTransform()
transform.extract_measures(X_in, "./output.csv")
```

::: photonai_graph.Controllability.controllability_measures.ControllabilityMeasureTransform.extract_measures
