---
layout:     page
title:
permalink:  /
---

<br>

## Posts

<br>

<div class="row">
    <div class="col-xs-12">
        <ul class="entries">
            {% for post in site.posts %}
                <li>
                    <span class="title"><a href="{{ post.url }}">{{ post.title }}</a></span>
                    <span class="date">{{ post.date | date: "%d %B %Y" }}</span>
                </li>
            {% endfor %}
        </ul>
    </div>
</div>
