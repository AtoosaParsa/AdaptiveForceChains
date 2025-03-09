function [im S]=peipf2(cr,x,y,P,Q)
% peipf    Calculate photoelastic image from forces PQ at D/2, phi. 
% Usage: im=peipf(cr,PQ,phi,D)
%
% Calculates a photoelatic image from normal P and tangential Q forces where 
% PQ=[P(1:N) Q(1:N)] applied at a radius of D(1:N)/2 and angle phi(1:N). The
% origin is where abs(cr)==0.

% revision history:
% 02/03/93 Mark D. Shattuck <mds> peipf.m
%          mds convert from from peipf.c
% 01/30/14 mds update for book

Nf=length(P);

S=0;
for nf=1:Nf
  S=S+PointForceHPrt(cr,x(nf),y(nf),P(nf),Q(nf));
end

im=Tmax(S);
