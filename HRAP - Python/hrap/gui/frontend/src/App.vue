<script setup lang="ts">
import HelloWorld from './components/HelloWorld.vue'
import TheWelcome from './components/TheWelcome.vue'
import Tank from './components/Tank.vue'

</script>

<template>
  <header>
    <img alt="Vue logo" class="logo" src="./assets/logo.svg" width="125" height="125" />

    <div class="wrapper">
      <HelloWorld msg="You did it!" />
    </div>
  </header>

  <!-- <main> -->
    <!-- <TheWelcome /> -->
  <!-- </main> -->
  <main>
    <Tank/>
  </main>
  
  <button class="btn btn-primary">Button</button>
  <input type="text" placeholder="Type here" class="input" />
  
  <input type="range" min="0" max="100" value="40" class="range" />
  
  <label class="input">
    <span class="label">Fill level</span>
    <input type="range" min="0" max="100" value="70" class="range" />
    <input type="text" placeholder="70%" />
  </label>
  
  <label class="select">
    <span class="label">Oxidizer</span>
    <select>
      <option>Nitrous Oxide</option>
      <option>Oxygen</option>
    </select>
  </label>
  
  <label class="input">
    <input type="text" placeholder="domain name" />
    <span class="label">.com</span>
  </label>
  
  <div class="skeleton h-32 w-32"></div>
</template>

<!-- <style scoped> -->
<!-- header { -->
  <!-- line-height: 1.5; -->
<!-- } -->

<!-- .logo { -->
  <!-- display: block; -->
  <!-- margin: 0 auto 2rem; -->
<!-- } -->

<!-- @media (min-width: 1024px) { -->
  <!-- header { -->
    <!-- display: flex; -->
    <!-- place-items: center; -->
    <!-- padding-right: calc(var(--section-gap) / 2); -->
  <!-- } -->

  <!-- .logo { -->
    <!-- margin: 0 2rem 0 0; -->
  <!-- } -->

  <!-- header .wrapper { -->
    <!-- display: flex; -->
    <!-- place-items: flex-start; -->
    <!-- flex-wrap: wrap; -->
  <!-- } -->
<!-- } -->
<!-- </style> -->
