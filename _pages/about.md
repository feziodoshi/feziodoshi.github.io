---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi, I am a Ph.D. candidate in the Dept. of Psychology at Harvard University, advised by [Prof. George Alvarez](https://visionlab.harvard.edu/george/bio) and [Prof. Talia Konkle](https://konklab.fas.harvard.edu/#) in the [Vision Sciences Lab](https://visionlab.harvard.edu) and [Cognitive and Neural Organization Lab](https://konklab.fas.harvard.edu/#). I am grateful to be supported by the Graduate Fellowship at the [Kempner Institute for the Study of Natural and Artificial Vision](https://kempnerinstitute.harvard.edu/). Broadly, I study how the human visual system transforms raw sensory input into structured, meaningful representations and how we can uncover similar mechanisms inside large-scale vision models.

<!-- > My work bridges ***cognitive science, computational vision, and mechanistic interpretability*** to understand mid-level vision: how local elements are grouped into proto-objects, how global shape emerges, and how long-range interactions support object perception in both humans and artificial systems. -->

<blockquote class="hover-bridge">
  My work bridges <strong><em>cognitive science, computational vision, and mechanistic interpretability</em></strong> 
  to understand mid-level vision: how local elements are grouped into proto-objects, how global shape emerges, 
  and how long-range interactions support object perception in both humans and artificial systems.
</blockquote>

* **On the cognitive and computational side,** I study whether self-supervised learning, distillation, and vision-language alignment provide the inductive biases needed for models to learn holistic shape and show human-like sensitivity to configural relations between object parts.

* **On the mechanistic side,** we build a decomposition framework that separates positional and semantic information in Vision Transformers via spectral factorization of the query–key interaction, allowing us to isolate the circuits that govern internal information flow.

* **On the brain and behavior side,** I examine the tuning and spatial organization of visual representations, characterizing how mid-level perceptual groupng mechanisms give rise to object-shape perception in humans and vision models.

Across these projects, my goal is to combine theory-driven experiments with mechanistic analyses to uncover the computational principles that underlie both biological and artificial vision. For a brief overview of my ongoing research, you can read the short feature article linked on [this](https://kempnerinstitute.harvard.edu/news/ai-vision-models-can-mimic-a-key-step-in-how-the-human-brain-perceives-shapes/) and [this]((https://gsas.harvard.edu/news/seeing-how-we-see)) page, and learn more about my background and earlier work through the [Harvard Brain Science Initiative page](https://brain.harvard.edu/hbi_humans/fenil-rakesh-doshi/).


### Here is our recent work on holistic processing and shape perception in vision models
Link to [paper](https://arxiv.org/abs/2507.00493) (NeurIPS 2025), [project page](https://www.fenildoshi.com/configural-shape/), and [tweet-thread](https://x.com/fenildoshi009/status/1968807305811492940)

<img src="https://feziodoshi.github.io/images/configural_shape_holistic_paper.png" alt="drawing" style="width:100%;height: auto;display: block;margin-left: auto;margin-right: auto; border: 3px solid black;"/>


### Check out our work on modeling cortical topographies
Link to [paper](https://www.science.org/doi/10.1126/sciadv.ade8187) (Science Advances 2023) and [tweet-thread](https://x.com/fenildoshi009/status/1567956934971768832)

<img src="https://feziodoshi.github.io/images/cover_cortical_topographies.png" alt="drawing" style="width:100%;height: auto;display: block;margin-left: auto;margin-right: auto; border: 3px solid black;"/>

### Here's some recent work on computational mechanims underlying mid-level object perception
Link to [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013391) (PLOS Computional Biology, 2025) and [tweet-thread](https://x.com/fenildoshi009/status/1960040583961223229?s=61)

<img src="https://feziodoshi.github.io/images/cover_contour_integration_dnns.png" alt="drawing" style="width:100%;height: auto;display: block;margin-left: auto;margin-right: auto; border: 3px solid black;"/>

### Talk on perceptual signatures of contour integration in humans and deep neural networks 
Presented at Vision Sciences Society 2022
{% include youtubePlayer.html id="PsmZAMGeV6A" %}

### Talk on a topographic model of the human visual system using self-organizing constraints
Presented at Vision Sciences Society 2021
{% include youtubePlayer.html id="zZvrIuoxU6Y" %}



<style>
/* subtle hover for the “bridges cognitive science…” block */
blockquote.hover-bridge {
  border-left: 3px solid #ddd;
  padding: 0.75rem 1rem;
  margin: 1.25rem 0;
  transition: background-color 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
}

blockquote.hover-bridge:hover {
  background-color: rgba(0, 0, 0, 0.02);
  border-color: #555;
  transform: translateY(-1px);
}

/* optional: if you ever want to tweak all paper images at once */
.paper-image {
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.paper-image:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
</style>