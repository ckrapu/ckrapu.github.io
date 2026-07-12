---
layout: about
title: about
nav: false
permalink: /
subtitle: 

profile:
  align: right
  image: 
  image_circular: false # crops the image to make it circular
  more_info: 

news: false # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false # includes social icons at the bottom of the page
---
<div class="about-intro">
<div class="about-intro__content" markdown="1">

<p class="about-location"><i class="ti ti-map-pin" aria-hidden="true"></i><span>bay area / north dakota</span></p>

Currently working on internal AI tools @ NVIDIA. 

I received my graduate training in statistics and engineering at Duke University under Mark Borsuk, graduating in 2020. I was awarded full-funding fellowships by the NSF ([IGERT, 2014](https://czo-archive.criticalzone.org/calhoun/news/story/duke-phd-students-win-2-year-igert-fellowships/)) and NASA ([NESSF, 2017](https://cce.nasa.gov/cgi-bin/terrestrial_ecology/pi_list.pl?projType=project&progID=2&projID=4037)) with additional support from Amazon and NVIDIA. My research focused on developing graphical models for high-dimensional spatial data, with applications in engineering, ecology, and the environment.


{% capture professional_info %}
- I spent six years in grad school. I loved grad school so much that I pushed back my defense for a year just so I could use all of my funding and focus on my research.
- I moved to NVIDIA in 2024, focusing on retrieval and agentic workflows for our sales & field operations.
{% endcapture %}
{% include collapsible.liquid title="Professional experience" content=professional_info %}

{% capture personal_info %}
- I was hard of hearing in my early life, and spent most of my childhood in speech therapy.
- I started college at Valley City State University and later studied at Macalester College for physics. 
- My favorite physical system is the spin glass.
- My family name is of Finnish origin. My great-grandfather emigrated from Finland around 1890.
- You can email me using the format of first initial followed by last name (no punctuation) at gmail

{% endcapture %}

{% include collapsible.liquid title="Personal trivia" content=personal_info %}

</div>
<div class="about-intro__visual">
{% include ising_sampler.liquid %}
</div>
</div>
