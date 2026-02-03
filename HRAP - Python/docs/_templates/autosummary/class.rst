{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. py:class:: {{ objname }}

   .. TODO: when no new
   .. .. automethod:: {{ objname }}.__init__

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   {% for item in methods %}
      {% if item != "__init__" %}
   .. automethod:: {{ objname }}.{{ item }}
      {% endif %}
   {% endfor %}

   {% endif %}
   {% endblock %}

   .. {% block attributes_og %}
   .. {% if attributes %}
   .. .. rubric:: {{ _('Attributes') }}

   .. .. autosummary::
   .. {% for item in attributes %}
   ..    ~{{ name }}.{{ item }}
   .. {%- endfor %}
   .. {% endif %}
   .. {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
      :annotation:
   {% endfor %}

   {% endif %}
   {% endblock %}
