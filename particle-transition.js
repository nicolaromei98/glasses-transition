/**
 * ParticleTransition — GPU particle dissolve transition between 3D objects.
 *
 * Reusable, framework-agnostic Three.js module. Zero dependencies beyond
 * three + two official addons.
 *
 * USAGE:
 *
 *   import { ParticleTransition } from './particle-transition.js';
 *
 *   // `models`: array of THREE.Object3D already added to the SAME parent
 *   // (a Group or the Scene). Position/scale them however you want first.
 *   const pt = new ParticleTransition({
 *     models: [modelA, modelB, modelC],
 *     parent: stageGroup,            // common parent (default: models[0].parent)
 *     accents: [0x00e5ff, 0xff3df0], // one color per model (edge glow + particles)
 *     particleCount: 30000,
 *     duration: 2600,                // ms
 *     particleSize: 0.05,            // tune to your scene scale
 *     noiseScale: 2.2,               // dissolve pattern frequency (lower = bigger chunks)
 *     onProgress: p => {},           // 0..1 each frame during a transition (e.g. pump bloom)
 *   });
 *
 *   pt.showOnly(0);                  // initial state
 *
 *   // in your render loop, BEFORE renderer.render():
 *   pt.update();
 *
 *   // trigger (returns a Promise that resolves when done):
 *   await pt.transitionTo(1);
 *
 *   pt.dispose();                    // cleanup
 *
 * NOTES:
 * - Particles fly in the parent's local space, so a rotating parent group
 *   rotates the whole effect coherently.
 * - Works with any built-in material (Standard/Physical/Phong...), including
 *   transmission. The dissolve is injected via onBeforeCompile.
 * - Pairs well with UnrealBloomPass: edge glow is pushed to ~5x emissive.
 */

import * as THREE from 'three';
import { MeshSurfaceSampler } from 'three/addons/math/MeshSurfaceSampler.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';

/* ---------------- GLSL ---------------- */

const NOISE_GLSL = /* glsl */`
vec3 ptMod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 ptMod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 ptPermute(vec4 x){return ptMod289(((x*34.0)+1.0)*x);}
vec4 ptTaylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float ptSnoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=ptMod289(i);
  vec4 p=ptPermute(ptPermute(ptPermute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=ptTaylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
vec3 ptCurl(vec3 p){
  return vec3(
    ptSnoise(p+vec3(31.4,0.0,0.0)),
    ptSnoise(p+vec3(0.0,47.2,0.0)),
    ptSnoise(p+vec3(0.0,0.0,12.9))
  );
}`;

const PARTICLE_VERT = /* glsl */`
uniform float uProgress;
uniform float uTime;
uniform float uSize;
uniform float uSwirl;
attribute vec3 aFrom;
attribute vec3 aTo;
attribute vec4 aRand;
varying float vMid;
varying float vT;
varying float vRand;
${NOISE_GLSL}
void main(){
  float t = clamp((uProgress - aRand.w * 0.3) / 0.7, 0.0, 1.0); // staggered start
  float e = t * t * (3.0 - 2.0 * t);
  vec3 p = mix(aFrom, aTo, e);

  float mid = sin(t * 3.14159265);
  vec3 swirl = ptCurl(aFrom * 1.6 + aRand.xyz * 6.0 + uTime * 0.15);
  vec3 tangent = normalize(vec3(-p.z, 0.35 * (aRand.y - 0.5), p.x) + 0.0001);
  p += swirl * mid * (0.45 + aRand.x * 0.9) * uSwirl;
  p += tangent * mid * (0.5 + aRand.z * 0.8) * sign(aRand.y - 0.5) * uSwirl;
  p.y += mid * 0.25 * (aRand.z - 0.3) * uSwirl;

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float sz = uSize * (0.6 + aRand.y * 1.4) * (mid + 0.001);
  gl_PointSize = sz * (300.0 / -mv.z);
  vMid = mid; vT = e; vRand = aRand.x;
}`;

const PARTICLE_FRAG = /* glsl */`
uniform vec3 uColorA;
uniform vec3 uColorB;
varying float vMid;
varying float vT;
varying float vRand;
void main(){
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c);
  if(d > 0.5) discard;
  float core = smoothstep(0.5, 0.05, d);
  vec3 col = mix(uColorA, uColorB, vT);
  col += vec3(0.6) * pow(core, 4.0) * vRand;
  float a = core * vMid;
  gl_FragColor = vec4(col * (1.0 + vMid * 2.5), a);
}`;

/* ---------------- class ---------------- */

export class ParticleTransition {

  constructor({
    models,
    parent = null,
    accents = [],
    particleCount = 30000,
    duration = 2600,
    particleSize = 0.05,
    noiseScale = 2.2,
    edgeWidth = 0.12,
    swirl = 1.0,
    onProgress = null,
  } = {}) {
    if (!models || models.length < 2) throw new Error('ParticleTransition: need at least 2 models');
    this.parent = parent || models[0].parent;
    if (!this.parent) throw new Error('ParticleTransition: models must be added to a parent first');

    this.duration = duration;
    this.noiseScale = noiseScale;
    this.edgeWidth = edgeWidth;
    this.onProgress = onProgress;
    this.current = 0;
    this._shift = null;
    this._startTime = performance.now();

    // per-model entries
    this.entries = models.map((model, i) => {
      const accent = new THREE.Color(accents[i] !== undefined ? accents[i] : 0x00e5ff);
      const uniforms = {
        uDissolve: { value: 1 },
        uEdgeColor: { value: accent },
      };
      this._patchDissolve(model, uniforms);
      const points = this._samplePoints(model, particleCount);
      model.visible = false;
      return { model, points, uniforms, accent };
    });

    // particle system
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(particleCount * 3), 3));
    this._aFrom = new THREE.BufferAttribute(new Float32Array(particleCount * 3), 3);
    this._aTo   = new THREE.BufferAttribute(new Float32Array(particleCount * 3), 3);
    const aRand = new THREE.BufferAttribute(new Float32Array(particleCount * 4), 4);
    for (let i = 0; i < particleCount * 4; i++) aRand.array[i] = Math.random();
    geo.setAttribute('aFrom', this._aFrom);
    geo.setAttribute('aTo', this._aTo);
    geo.setAttribute('aRand', aRand);

    this._pUniforms = {
      uProgress: { value: 0 },
      uTime:     { value: 0 },
      uSize:     { value: particleSize },
      uSwirl:    { value: swirl },
      uColorA:   { value: new THREE.Color() },
      uColorB:   { value: new THREE.Color() },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this._pUniforms,
      vertexShader: PARTICLE_VERT,
      fragmentShader: PARTICLE_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.particles = new THREE.Points(geo, mat);
    this.particles.visible = false;
    this.particles.frustumCulled = false;
    this.parent.add(this.particles);
  }

  /* ---- public API ---- */

  /** Show one model instantly (initial state). */
  showOnly(index) {
    this.entries.forEach((e, i) => {
      e.model.visible = i === index;
      e.uniforms.uDissolve.value = i === index ? 0 : 1;
    });
    this.current = index;
  }

  /** Animate to model `index`. Resolves when the transition completes. */
  transitionTo(index) {
    if (this._shift || index === this.current) return Promise.resolve(false);
    index = ((index % this.entries.length) + this.entries.length) % this.entries.length;

    const from = this.entries[this.current];
    const to = this.entries[index];

    this._aFrom.array.set(from.points); this._aFrom.needsUpdate = true;
    this._aTo.array.set(to.points);     this._aTo.needsUpdate = true;
    this._pUniforms.uColorA.value.copy(from.accent);
    this._pUniforms.uColorB.value.copy(to.accent);

    to.uniforms.uDissolve.value = 1;
    to.model.visible = false;
    this.particles.visible = true;
    this._pUniforms.uProgress.value = 0;

    return new Promise(resolve => {
      this._shift = { from, to, target: index, start: performance.now(), resolve };
    });
  }

  get isTransitioning() { return !!this._shift; }

  /** Call once per frame, before rendering. */
  update() {
    const now = performance.now();
    this._pUniforms.uTime.value = (now - this._startTime) / 1000;
    if (!this._shift) return;

    const s = this._shift;
    const p = Math.min((now - s.start) / this.duration, 1);

    // outgoing mesh dissolves during 0 .. 0.42
    s.from.uniforms.uDissolve.value = THREE.MathUtils.clamp(p / 0.42, 0, 1);
    if (p > 0.46) s.from.model.visible = false;

    // particles fly during 0.05 .. 0.95
    this._pUniforms.uProgress.value = THREE.MathUtils.clamp((p - 0.05) / 0.9, 0, 1);

    // incoming mesh materializes during 0.55 .. 1
    if (p > 0.5) {
      s.to.model.visible = true;
      s.to.uniforms.uDissolve.value = 1 - THREE.MathUtils.clamp((p - 0.55) / 0.45, 0, 1);
    }

    if (this.onProgress) this.onProgress(p);

    if (p >= 1) {
      s.to.uniforms.uDissolve.value = 0;
      this.particles.visible = false;
      this.current = s.target;
      this._shift = null;
      s.resolve(true);
    }
  }

  dispose() {
    this.parent.remove(this.particles);
    this.particles.geometry.dispose();
    this.particles.material.dispose();
    this.entries.forEach(e => {
      e.model.traverse(o => {
        if (!o.isMesh || !o.material) return;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(m => { m.onBeforeCompile = () => {}; m.needsUpdate = true; });
      });
    });
  }

  /* ---- internals ---- */

  /** Sample `count` surface points in the parent's local space. */
  _samplePoints(object, count) {
    this.parent.updateMatrixWorld(true);
    object.updateMatrixWorld(true);
    const parentInv = new THREE.Matrix4().copy(this.parent.matrixWorld).invert();
    const rel = new THREE.Matrix4();
    const geos = [];
    object.traverse(o => {
      if (o.isMesh && o.geometry) {
        let g = o.geometry.clone();
        if (g.index) g = g.toNonIndexed();
        rel.multiplyMatrices(parentInv, o.matrixWorld);
        g.applyMatrix4(rel);
        for (const k of Object.keys(g.attributes)) if (k !== 'position') g.deleteAttribute(k);
        g.morphAttributes = {};
        geos.push(g);
      }
    });
    const merged = BufferGeometryUtils.mergeGeometries(geos, false);
    const sampler = new MeshSurfaceSampler(new THREE.Mesh(merged)).build();
    const arr = new Float32Array(count * 3);
    const v = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      sampler.sample(v);
      arr[i * 3] = v.x; arr[i * 3 + 1] = v.y; arr[i * 3 + 2] = v.z;
    }
    geos.forEach(g => g.dispose());
    merged.dispose();
    return arr;
  }

  /** Inject noise-dissolve + emissive edge into every material of the object. */
  _patchDissolve(object, uniforms) {
    const seen = new Set();
    const noiseScale = this.noiseScale;
    const edgeWidth = this.edgeWidth;
    object.traverse(o => {
      if (!o.isMesh || !o.material) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach(mat => {
        if (seen.has(mat)) return;
        seen.add(mat);
        mat.onBeforeCompile = (shader) => {
          shader.uniforms.uDissolve = uniforms.uDissolve;
          shader.uniforms.uEdgeColor = uniforms.uEdgeColor;
          shader.vertexShader = shader.vertexShader
            .replace('#include <common>', '#include <common>\nvarying vec3 vDissolveP;')
            .replace('#include <begin_vertex>', '#include <begin_vertex>\nvDissolveP = (modelMatrix * vec4(transformed, 1.0)).xyz;');
          shader.fragmentShader = shader.fragmentShader
            .replace('#include <common>',
              '#include <common>\nuniform float uDissolve;\nuniform vec3 uEdgeColor;\nvarying vec3 vDissolveP;\n' + NOISE_GLSL)
            .replace('#include <dithering_fragment>',
              `#include <dithering_fragment>
              if (uDissolve > 0.0001) {
                float dn = ptSnoise(vDissolveP * ${noiseScale.toFixed(3)}) * 0.5 + 0.5;
                float th = uDissolve * 1.15;
                if (dn < th) discard;
                float edge = 1.0 - smoothstep(th, th + ${edgeWidth.toFixed(3)}, dn);
                gl_FragColor.rgb = mix(gl_FragColor.rgb, uEdgeColor * 5.0, edge);
              }`);
        };
        mat.needsUpdate = true;
      });
    });
  }
}
