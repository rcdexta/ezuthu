(window.webpackJsonp=window.webpackJsonp||[]).push([[0],{27:function(e,t,a){e.exports=a(65)},32:function(e,t,a){},65:function(e,t,a){"use strict";a.r(t);var n=a(0),l=a.n(n),r=a(21),o=a.n(r),s=(a(32),a(26)),i=a(25),c=a.n(i),m=a(3);const d=(e,t)=>{const a=e.split(","),n=a[0].match(/:(.*?);/)[1],l=atob(a[1]);let r=l.length;const o=new Uint8Array(r);for(;r;)o[r-1]=l.charCodeAt(r-1),r-=1;return new File([o],t,{type:n})};class u extends l.a.Component{constructor(e){super(e),this.state={ranks:void 0},this.clear=(()=>{this.saveableCanvas.clear(),this.setState({ranks:void 0})}),this.saveImage=(()=>{const e=this.saveableCanvas.canvas.drawing;let t,a=e.getContext("2d"),n=e.width,l=e.height;t=a.getImageData(0,0,n,l);var r=a.globalCompositeOperation;a.globalCompositeOperation="destination-over",a.fillStyle="#fff",a.fillRect(0,0,n,l);var o=e.toDataURL("image/".concat("png"));a.clearRect(0,0,n,l),a.putImageData(t,0,0),a.globalCompositeOperation=r;const s=d(o),i=new FormData;i.append("file",s,s.name);c.a.post("/upload",i,{headers:{"Content-Type":"multipart/form-data"}}).then(e=>{const t=e.data;this.setState({ranks:t}),window.scrollTo(0,document.body.scrollHeight)})}),this.saveableCanvas=l.a.createRef()}render(){const e=this.state.ranks;return l.a.createElement("div",{className:"App"},l.a.createElement("h2",null,"Tamil Character Classification"),l.a.createElement(m.Container,null,l.a.createElement(m.Row,null,l.a.createElement(m.Col,{sm:6},"Draw a tamil alphabet filling the canvas below",l.a.createElement("div",{style:{border:"1px solid #000",marginTop:5}},l.a.createElement(s.a,{ref:e=>this.saveableCanvas=e,lazyRadius:0,brushColor:"#000",canvasWidth:"100%"})),l.a.createElement("button",{onClick:this.clear},"Clear"),l.a.createElement("button",{onClick:()=>{this.saveableCanvas.undo()}},"Undo"),l.a.createElement("button",{onClick:this.saveImage},"Submit")),l.a.createElement(m.Col,{sm:6},l.a.createElement(m.Container,null,l.a.createElement(m.Row,null,l.a.createElement(m.Col,null,e&&l.a.createElement("div",{style:{margin:"0.5em"}},l.a.createElement("div",{style:{fontSize:"4em"}},e.rank1),l.a.createElement("div",{style:{fontSize:11,marginTop:3}},"Guess 1")))),l.a.createElement(m.Row,null,l.a.createElement(m.Col,null,e&&l.a.createElement("div",{style:{margin:"0.5em"}},l.a.createElement("div",{style:{fontSize:"4em"}},e.rank2),l.a.createElement("div",{style:{fontSize:11,marginTop:3}},"Guess 2")))),l.a.createElement(m.Row,null,l.a.createElement(m.Col,null,e&&l.a.createElement("div",{style:{margin:"0.5em"}},l.a.createElement("div",{style:{fontSize:"4em"}},e.rank3),l.a.createElement("div",{style:{fontSize:11,marginTop:3}},"Guess 3")))))))))}}const p=document.getElementById("root");o.a.render(l.a.createElement(l.a.StrictMode,null,l.a.createElement(u,null)),p)}},[[27,1,2]]]);
//# sourceMappingURL=main.6e449740.chunk.js.map