# dateload
- timecode to frame
- calculate frame sampling in video
     * max time steps
     * frame count
- data process
    * enhancement
    * normalization
    * filling
- clarify
  > cross entropy loss
- mark
  > MSE loss
---

# STConv

```mermaid
flowchart TD
    A[video] --> B[TemporalConv]
    B --> C[CheConv]
    C --> D[TemporalConv]
    D --> E[score]
```


    
---

# STAttention
---

# STGCNMultiTask

1. Predictive classification
   first attention
   feature fusion
   > 8 angle
   > 64 characteristic
3. mark
   second attention
   sigmoid
